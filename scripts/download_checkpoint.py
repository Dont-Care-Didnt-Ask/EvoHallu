"""Download a pinned HF checkpoint without an additional cache copy.

Uses only the standard library, streams weights, checks published LFS hashes,
and writes a provenance manifest. Useful on pods with a small root disk.
"""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("destination", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--spill-dir", type=Path)
    ap.add_argument("--prefer-spill-weights", action="store_true", help="Use spill storage first for weights, reserving shared memory for feature caches")
    ap.add_argument("--revision")
    ap.add_argument("--weight-format", choices=["all", "hf", "mistral"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    rev_path = "/revision/" + urllib.parse.quote(args.revision, safe="") if args.revision else ""
    metadata = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{args.model}{rev_path}?blobs=true", timeout=60))
    revision = metadata["sha"]
    extensions = (".safetensors", ".json", ".jinja", ".txt", ".model", ".tiktoken", ".py")
    files = [f for f in metadata["siblings"] if f["rfilename"].endswith(extensions)]
    # Exclude nested alternative formats and unrelated assets.
    files = [f for f in files if "/" not in f["rfilename"]]
    if args.weight_format == "hf":
        files = [f for f in files if not f["rfilename"].startswith("consolidated")]
    elif args.weight_format == "mistral":
        files = [f for f in files if not (f["rfilename"].startswith("model") and
                                          (f["rfilename"].endswith(".safetensors") or f["rfilename"].endswith(".index.json")))]
    if args.weight_format != "all":
        index = "model.safetensors.index.json" if args.weight_format == "hf" else "consolidated.safetensors.index.json"
        index_data = json.load(urllib.request.urlopen(f"https://huggingface.co/{args.model}/resolve/{revision}/{index}", timeout=60))
        required = set(index_data["weight_map"].values())
        selected = {f["rfilename"] for f in files if f["rfilename"].endswith(".safetensors")}
        if selected != required:
            raise ValueError("Selected weight files do not match the published shard index")
    if args.dry_run:
        print(json.dumps({"model": args.model, "revision": revision, "weight_format": args.weight_format,
                          "selected_weight_bytes": sum(f.get("size", 0) for f in files if f["rfilename"].endswith(".safetensors")),
                          "files": [{"file": f["rfilename"], "bytes": f.get("size")} for f in files]}))
        return
    remaining = sum(f.get("size", 0) for f in files if not (args.destination / f["rfilename"]).exists())
    free = shutil.disk_usage(args.destination).free - 1024**3
    spill_free = 0
    if args.spill_dir:
        args.spill_dir.mkdir(parents=True, exist_ok=True)
        spill_free = shutil.disk_usage(args.spill_dir).free - 1024**3
    storage = {}
    for entry in files:
        target = args.destination / entry["rfilename"]
        if target.exists() and target.stat().st_size == entry.get("size"):
            storage[entry["rfilename"]] = target
            continue
        size = entry.get("size", 0)
        if args.prefer_spill_weights and args.spill_dir and entry["rfilename"].endswith(".safetensors") and size <= spill_free:
            storage[entry["rfilename"]] = args.spill_dir / entry["rfilename"]
            spill_free -= size
        elif size <= free:
            storage[entry["rfilename"]] = target
            free -= size
        elif args.spill_dir and size <= spill_free:
            storage[entry["rfilename"]] = args.spill_dir / entry["rfilename"]
            spill_free -= size
        else:
            raise RuntimeError(f"Insufficient checkpoint storage for {entry['rfilename']}")
    (args.destination / "download_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"model": args.model, "revision": revision, "remaining_bytes": remaining}), flush=True)

    def download(entry):
        name = entry["rfilename"]
        public_path = args.destination / name
        dest = storage[name]
        expected_size = entry.get("size")
        expected_hash = entry.get("lfs", {}).get("sha256")
        if dest.exists() and dest.stat().st_size == expected_size:
            return {"file": name, "bytes": expected_size, "cached": True}
        url = f"https://huggingface.co/{args.model}/resolve/{revision}/{urllib.parse.quote(name)}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(3):
            digest = hashlib.sha256()
            try:
                with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as stream:
                    while chunk := response.read(8 * 1024**2):
                        stream.write(chunk)
                        digest.update(chunk)
                actual = tmp.stat().st_size
                if expected_size is not None and actual != expected_size:
                    raise ValueError(f"Size mismatch for {name}: {actual} != {expected_size}")
                if expected_hash and digest.hexdigest() != expected_hash:
                    raise ValueError(f"SHA256 mismatch for {name}")
                tmp.replace(dest)
                if dest != public_path:
                    public_path.symlink_to(dest)
                result = {"file": name, "bytes": actual, "sha256": digest.hexdigest()}
                print(json.dumps(result), flush=True)
                return result
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(download, files))
    (args.destination / "download_manifest.json").write_text(json.dumps({
        "model": args.model, "revision": revision, "weight_format": args.weight_format, "files": results,
        "prefer_spill_weights": args.prefer_spill_weights,
    }, indent=2))
    print("DOWNLOAD COMPLETE", flush=True)


if __name__ == "__main__":
    main()
