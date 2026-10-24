"""Restore audited feature archives as they arrive, without GPU inference."""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["mod1-eval-cache", "mod1-transfer-cache"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    manifest = json.loads((ROOT / "data/cache_backup_manifest.json").read_text())
    selected = [(ROOT / path, info) for path, info in manifest.items()
                if Path(path).parent.name in args.groups]
    output = ROOT / "results/overnight20261021/restored_caches.json"
    destination = ROOT / "data/restored"
    stop_waiting = time.monotonic() + 5400

    def restore(item):
        archive, info = item
        target = destination / archive.parent.name / archive.name.removesuffix(".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        while not archive.exists():
            if time.monotonic() > stop_waiting:
                raise TimeoutError(f"Archive did not arrive: {archive.name}")
            time.sleep(10)
        digest = hashlib.sha256()
        with archive.open("rb") as f:
            for block in iter(lambda: f.read(16 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != info["gzip_sha256"]:
            raise ValueError(f"Compressed archive SHA mismatch: {archive.name}")
        raw_digest = hashlib.sha256()
        temp = target.with_suffix(".part")
        with gzip.open(archive, "rb") as source, temp.open("wb") as dest:
            for block in iter(lambda: source.read(16 * 1024**2), b""):
                dest.write(block)
                raw_digest.update(block)
        if temp.stat().st_size != info["raw_bytes"]:
            raise ValueError(f"Raw size mismatch: {archive.name}")
        temp.replace(target)
        record = dict(archive=str(archive.relative_to(ROOT)), target=str(target.relative_to(ROOT)),
                      compressed_sha256=digest.hexdigest(), raw_sha256=raw_digest.hexdigest(),
                      raw_bytes=target.stat().st_size, utc=dt.datetime.now(dt.timezone.utc).isoformat())
        print(json.dumps(record), flush=True)
        return record

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(restore, item) for item in selected]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
