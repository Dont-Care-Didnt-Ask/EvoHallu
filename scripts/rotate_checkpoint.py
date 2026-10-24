"""Remove only downloaded weight shards, preserving provenance and tokenizers."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("directory", type=Path)
args = ap.parse_args()
manifest = args.directory / "download_manifest.json"
if not manifest.is_file():
    raise ValueError(f"No download provenance at {manifest}")
allowed = (Path("/dev/shm/mod1-models"), Path("/workspace/mod1-models"))
directory = args.directory.resolve()
if not any(directory.is_relative_to(base) for base in allowed):
    raise ValueError("Refusing to rotate a directory outside this campaign's model staging areas")
total = 0
for row in json.loads(manifest.read_text())["files"]:
    if not row["file"].endswith(".safetensors"):
        continue
    path = args.directory / row["file"]
    if path.is_symlink():
        target = path.resolve()
        if not target.is_relative_to(Path("/workspace/mod1-model-spill")):
            raise ValueError(f"Unexpected shard target: {target}")
        if target.exists():
            total += target.stat().st_size
            target.unlink()
        path.unlink()
    elif path.exists():
        total += path.stat().st_size
        path.unlink()
print(json.dumps({"model_directory": str(args.directory), "released_bytes": total}))
