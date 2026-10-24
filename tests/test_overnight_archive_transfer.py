import datetime as dt
import importlib.util
import json
import lzma
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overnight_archive_transfer", ROOT / "scripts/overnight_archive_transfer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_synthetic_roundtrip_preserves_all_raws_and_reuses_verified_archives(tmp_path, monkeypatch, capsys):
    rows, original = [], {}
    for index, (name, sample_count) in enumerate(module.EXPECTED.items()):
        path = tmp_path / name
        content = bytes(range(256)) * 500 + (f"synthetic cache {index}\n".encode() * 1000)
        path.write_bytes(content)
        original[path] = content
        digest, size = module.hash_file(path)
        rows.append(dict(path=str(path), raw_sha256=digest, raw_bytes=size, sample_count=sample_count))
    receipt = tmp_path / "completion.json"
    receipt.write_text(json.dumps(dict(status="complete_three_missing_caches",
        target=dict(revision=module.TARGET_REVISION), new_caches=rows)))
    output, manifest = tmp_path / "archives", tmp_path / "manifest.json"
    deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)).isoformat()
    command = ["archive", "--receipt", str(receipt), "--output-dir", str(output),
               "--manifest", str(manifest), "--deadline-utc", deadline, "--execute", "--workers", "2"]
    monkeypatch.setattr("sys.argv", command)
    module.main()
    capsys.readouterr()
    first = json.loads(manifest.read_text())
    assert first["status"] == "complete_three_new_cache_archives"
    assert len(first["archives"]) == 3
    mtimes = {}
    for row in first["archives"]:
        raw, archive = Path(row["source"]), Path(row["archive"])
        assert raw.read_bytes() == original[raw]
        assert lzma.open(archive, "rb").read() == original[raw]
        assert row["roundtrip_raw_sha256_verified"] and row["source_unchanged"]
        mtimes[archive] = archive.stat().st_mtime_ns
    module.main()
    capsys.readouterr()
    second = json.loads(manifest.read_text())
    assert all(row["reused_existing_archive"] for row in second["archives"])
    assert all(path.stat().st_mtime_ns == mtime for path, mtime in mtimes.items())
    assert all(path.read_bytes() == content for path, content in original.items())
