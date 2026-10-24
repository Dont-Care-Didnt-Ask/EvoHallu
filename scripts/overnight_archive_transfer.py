"""Losslessly archive only the three newly completed 1B feature caches.

Dry-run by default. Original caches are never removed or overwritten. Explicit
execution uses xz -T4 -6 (or a bounded Python-lzma subprocess), verifies the
round trip against completion-receipt hashes, and publishes atomic receipts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
HARD_DEADLINE = dt.datetime(2026, 10, 22, 2, 6, 0, tzinfo=dt.timezone.utc)
EXPECTED = {
    "geometry__counterfact_true_false.pt": 31964,
    "popqa__popqa_test_filtered_10shot.pt": 6974,
    "x-fact__test_filtered_10shot.pt": 2315,
}
TARGET_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"


def check_time(deadline, abort):
    if abort.is_set():
        raise concurrent.futures.CancelledError("Another cache archive failed")
    if dt.datetime.now(dt.timezone.utc) >= deadline:
        raise TimeoutError("Shared feature-archive deadline reached")


def hash_file(path, deadline=None, abort=None):
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            if deadline is not None:
                check_time(deadline, abort)
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compression_command(source, xz):
    if xz:
        # --stdout leaves the original file intact; -- handles unusual paths.
        return [xz, "-T4", "-6", "--stdout", "--", str(source)]
    code = ("import lzma,shutil,sys; "
            "source=open(sys.argv[1],'rb'); "
            "target=lzma.LZMAFile(sys.stdout.buffer,'wb',preset=6); "
            "shutil.copyfileobj(source,target,8*1024**2); target.close(); source.close()")
    return [sys.executable, "-c", code, str(source)]


def verify_roundtrip(archive, row, deadline, abort):
    digest, size = hashlib.sha256(), 0
    with lzma.open(archive, "rb") as stream:
        while block := stream.read(8 * 1024**2):
            check_time(deadline, abort)
            digest.update(block)
            size += len(block)
    if size != row["raw_bytes"] or digest.hexdigest() != row["raw_sha256"]:
        raise ValueError("Archive does not reproduce the pinned raw cache: " + str(archive))


def terminate(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def archive_one(row, output_dir, deadline, abort, xz, known=None):
    source = Path(row["path"])
    destination = output_dir / (source.name + ".xz")
    temporary = destination.with_suffix(destination.suffix + ".part")
    started = time.monotonic()
    check_time(deadline, abort)
    before_hash, before_size = hash_file(source, deadline, abort)
    if before_hash != row["raw_sha256"] or before_size != row["raw_bytes"]:
        raise ValueError("Source cache differs from its completion receipt: " + str(source))
    reused = destination.exists()
    command = compression_command(source, xz)
    if reused:
        archive_hash, archive_bytes = hash_file(destination, deadline, abort)
        if known and (known["archive_sha256"] != archive_hash or known["archive_bytes"] != archive_bytes or
                      known["raw_sha256"] != before_hash or known["raw_bytes"] != before_size):
            raise ValueError("Existing archive differs from its previous receipt; refusing overwrite: " + str(destination))
        # A valid archive can be adopted after interruption between rename and
        # receipt writing. Unknown/conflicting archives are never overwritten.
        verify_roundtrip(destination, row, deadline, abort)
    else:
        if temporary.exists():
            raise ValueError("Refusing to overwrite an unreviewed partial archive: " + str(temporary))
        with temporary.open("xb") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.PIPE,
                                     start_new_session=True)
            try:
                while child.poll() is None:
                    check_time(deadline, abort)
                    time.sleep(.1)
                if child.returncode:
                    error = child.stderr.read(4096).decode(errors="replace")
                    raise RuntimeError(f"Archive compressor failed ({child.returncode}): {error}")
            finally:
                terminate(child)
                child.stderr.close()
        verify_roundtrip(temporary, row, deadline, abort)
        archive_hash, archive_bytes = hash_file(temporary, deadline, abort)
    after_hash, after_size = hash_file(source, deadline, abort)
    if after_hash != before_hash or after_size != before_size:
        raise ValueError("Source cache changed during compression: " + str(source))
    if not reused:
        check_time(deadline, abort)
        if destination.exists():
            raise FileExistsError("An archive appeared during compression; refusing replacement")
        temporary.replace(destination)
    return dict(source=str(source), raw_sha256=before_hash, raw_bytes=before_size,
        archive=str(destination), archive_sha256=archive_hash, archive_bytes=archive_bytes,
        source_unchanged=True, roundtrip_raw_sha256_verified=True,
        reused_existing_archive=reused, elapsed_seconds=time.monotonic()-started,
        compression="xz -T4 -6" if xz else "Python lzma preset 6 subprocess", command=command)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, default=ROOT / "results/overnight20261021/remaining_transfer_completion.json")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/feature_cache_archive/overnight20261021")
    ap.add_argument("--manifest", type=Path, default=ROOT / "results/overnight20261021/new_feature_cache_archives.json")
    ap.add_argument("--workers", type=int, choices=(1, 2), default=2)
    ap.add_argument("--deadline-utc", default="2026-10-22T02:06:00Z")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline.utcoffset() != dt.timedelta(0) or deadline > HARD_DEADLINE:
        ap.error("Archive deadline must be UTC and no later than 2026-10-22T02:06:00Z")
    if not args.receipt.exists():
        if args.execute:
            raise FileNotFoundError("Complete the three target caches before archiving: " + str(args.receipt))
        print(json.dumps(dict(status="waiting_for_completed_transfer_receipt", receipt=str(args.receipt),
                              planned_files=list(EXPECTED), execute=False)))
        return
    receipt = json.loads(args.receipt.read_text())
    if receipt.get("status") != "complete_three_missing_caches" or receipt.get("target", {}).get("revision") != TARGET_REVISION:
        raise ValueError("Archiving requires the completed pinned 1B transfer receipt")
    rows = receipt.get("new_caches", [])
    if len(rows) != 3 or {Path(row["path"]).name for row in rows} != set(EXPECTED):
        raise ValueError("Archive exactly the three new 1B feature-cache files")
    if any(row["sample_count"] != EXPECTED[Path(row["path"]).name] for row in rows):
        raise ValueError("Unexpected sample counts in the transfer completion receipt")
    xz = shutil.which("xz")
    plan = dict(status="planned_not_executed", source_receipt=str(args.receipt),
        source_receipt_sha256=hash_file(args.receipt)[0], deadline_utc=deadline.isoformat(),
        output_directory=str(args.output_dir), workers=args.workers, xz_threads_per_file=4 if xz else 1,
        original_raw_files_removed=False, archives=[],
        planned=[dict(source=row["path"], raw_bytes=row["raw_bytes"], raw_sha256=row["raw_sha256"],
                      archive=str(args.output_dir / (Path(row["path"]).name + ".xz")),
                      command=compression_command(Path(row["path"]), xz)) for row in rows])
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    old = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    known = {row["source"]: row for row in old.get("archives", [])}
    if known and old.get("source_receipt_sha256") != plan["source_receipt_sha256"]:
        raise ValueError("Existing archive manifest belongs to a different completion receipt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    abort = threading.Event()
    state = {**plan, "status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    futures, recorded = [], set()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    old_handlers = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Archive work interrupted by signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, interrupted)

    def save(future):
        state["archives"].append(future.result())
        recorded.add(future)
        atomic_json(args.manifest, state)

    atomic_json(args.manifest, state)
    failure = None
    try:
        for row in rows:
            futures.append(pool.submit(archive_one, row, args.output_dir, deadline, abort, xz, known.get(row["path"])))
        for future in concurrent.futures.as_completed(futures):
            save(future)
        state["status"] = "complete_three_new_cache_archives"
    except BaseException as exc:
        failure = exc
        state.update(status="partial_or_failed", error=f"{type(exc).__name__}: {exc}")
        abort.set()
        raise
    finally:
        if failure is not None:
            abort.set()
        pool.shutdown(wait=True, cancel_futures=True)
        for future in futures:
            if future in recorded or future.cancelled():
                continue
            try:
                save(future)
            except BaseException:
                pass
        state["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        state["total_archive_bytes"] = sum(row["archive_bytes"] for row in state["archives"])
        atomic_json(args.manifest, state)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps(dict(status=state["status"], archives=len(state["archives"]),
                         total_archive_bytes=state["total_archive_bytes"], manifest=str(args.manifest))))


if __name__ == "__main__":
    main()
