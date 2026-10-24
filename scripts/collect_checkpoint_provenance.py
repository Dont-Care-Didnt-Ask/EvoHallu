"""Preserve small public checkpoint metadata before temporary weights rotate."""
import json
import shutil
from pathlib import Path


def main():
    output = Path("out/campaign/model_provenance")
    copied = []
    roots = [Path("/dev/shm/mod1-models"), Path("/workspace/mod1-models")]
    names = ["config.json", "tokenizer_config.json", "generation_config.json",
             "chat_template.jinja", "download_metadata.json", "download_manifest.json",
             "params.json", "special_tokens_map.json", "model.safetensors.index.json",
             "consolidated.safetensors.index.json"]
    for root in roots:
        for manifest in sorted(root.glob("*/download_manifest.json")):
            source = manifest.parent
            record = json.loads(manifest.read_text())
            destination = output / source.name
            destination.mkdir(parents=True, exist_ok=True)
            for name in names:
                path = source / name
                if path.exists():
                    # An idempotent download records cached files without their
                    # original computed hashes. Retain the first full manifest.
                    target = destination / name
                    if name == "download_manifest.json" and target.exists():
                        previous = json.loads(target.read_text())
                        if (previous.get("revision") == record.get("revision") and
                                sum("sha256" in f for f in previous["files"]) >
                                sum("sha256" in f for f in record["files"])):
                            continue
                    shutil.copy2(path, target)
            copied.append({"model": record["model"], "revision": record["revision"],
                           "destination": str(destination)})
    print(json.dumps(copied))


if __name__ == "__main__":
    main()
