"""Stream-check gzip CRC, byte counts and hashes of preserved raw feature caches."""
import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/feature_cache_archive"))
    parser.add_argument("--output", type=Path, default=Path("results/feature_cache_validation.json"))
    args = parser.parse_args()
    old = json.loads(args.output.read_text()).get("archives", {}) if args.output.exists() else {}
    expected = json.loads(Path("data/cache_backup_manifest.json").read_text())
    training_manifest = Path("results/training_feature_cache_manifest.json")
    if training_manifest.exists():
        training = json.loads(training_manifest.read_text())
        expected[training["gzip_file"]] = training
    records = {}
    for path in sorted(args.root.rglob("*.pt.gz")):
        key = str(path)
        info = path.stat()
        if (key in old and old[key].get("gzip_bytes") == info.st_size and old[key].get("mtime_ns") == info.st_mtime_ns
                and (old[key].get("backup_manifest_verified") or key not in expected)):
            records[key] = old[key]
            continue
        digest = hashlib.sha256()
        total = 0
        # Reading to EOF also verifies gzip's CRC and uncompressed-size trailer.
        with gzip.open(path, "rb") as stream:
            while block := stream.read(8*1024**2):
                digest.update(block)
                total += len(block)
        with path.open("rb") as stream:
            compressed_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        reference = expected.get(key, {})
        if reference and (total != reference["raw_bytes"] or info.st_size != reference["gzip_bytes"] or
                          compressed_digest != reference["gzip_sha256"]):
            raise ValueError("Feature archive differs from its backup manifest: " + key)
        if reference.get("raw_sha256") and digest.hexdigest() != reference["raw_sha256"]:
            raise ValueError("Uncompressed cache hash differs: " + key)
        records[key] = {"gzip_crc_valid": True, "raw_bytes": total, "raw_sha256": digest.hexdigest(),
                        "gzip_bytes": info.st_size, "gzip_sha256": compressed_digest, "mtime_ns": info.st_mtime_ns,
                        "backup_manifest_verified": bool(reference)}
        print(json.dumps({"verified": key, "raw_bytes": total}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "archives": records,
              "count": len(records), "all_gzip_crc_valid": all(r["gzip_crc_valid"] for r in records.values())}
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2)+"\n")
    tmp.replace(args.output)
    print(json.dumps({"verified_archives": len(records)}), flush=True)


if __name__ == "__main__":
    main()
