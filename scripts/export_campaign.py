"""Export committed source plus compact campaign artifacts, without raw caches."""
import argparse
import hashlib
import re
import subprocess
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.output.resolve().is_relative_to(root):
        raise SystemExit("Place the archive outside the repository")
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    extras = sorted(p for prefix in ["out/logs/mod1", "out/campaign"] for p in Path(prefix).rglob("*") if p.is_file())
    blocked = {".pt", ".safetensors", ".gz", ".part"}
    extras = [p for p in extras if p.suffix not in blocked]
    # Stored trial credentials are placeholders. Reject accidental recognizable
    # credential values without ever displaying the matching content.
    credential = re.compile(rb"(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{30,}|rpa_[A-Za-z0-9]{25,})")
    for path in [Path(name) for name in tracked if name] + extras:
        if not path.is_file():
            continue
        if path.stat().st_size > 512*1024**2:
            raise SystemExit("Unexpected large artifact: " + str(path))
        if path.suffix in {".json", ".jsonl", ".yaml", ".log", ".txt", ".py"} and credential.search(path.read_bytes()):
            raise SystemExit("Possible credential in artifact: " + str(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+".part")
    try:
        with temporary.open("wb") as stream:
            subprocess.run(["git", "archive", "--format=zip", "--prefix=mod1/", "HEAD"], stdout=stream, check=True)
        with zipfile.ZipFile(temporary, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in extras:
                archive.write(path, "mod1/"+path.as_posix())
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError("Archive CRC check failed: " + bad)
            members = len(archive.infolist())
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.file_digest(args.output.open("rb"), "sha256").hexdigest()
    args.output.with_suffix(args.output.suffix+".sha256").write_text(digest+"  "+args.output.name+"\n")
    print({"archive": str(args.output.resolve()), "bytes": args.output.stat().st_size, "files": members,
           "sha256": digest, "raw_feature_caches_included": False})


if __name__ == "__main__":
    main()
