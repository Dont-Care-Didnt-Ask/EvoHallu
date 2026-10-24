"""LOCAL-only watcher for the three new, losslessly archived transfer caches.

Dry-run by default: no SSH, launch, download, or local mutation. Explicit
--execute may start only the existing CPU archive helper, then download its
three verified archives. Never extracts features, deletes inputs, or stops a
pod. The independent resource guard remains responsible for pod shutdown.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/overnight20261021"
DESTINATION = ROOT / "data/feature_cache_archive/overnight20261021"
REMOTE_ROOT = "/workspace/mod1"
REMOTE_OUTPUT = REMOTE_ROOT + "/results/overnight20261021"
REMOTE_ARCHIVES = REMOTE_ROOT + "/data/feature_cache_archive/overnight20261021"
HOST = "root@204.12.201.57"
SSH = ["ssh", "-i", "/root/.ssh/processbench_pod_ed25519", "-p", "5933",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
       "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=accept-new"]
HARD_STOP = dt.datetime(2026,10,22,2,13,tzinfo=dt.timezone.utc)
ARCHIVE_CUTOFF = dt.datetime(2026,10,22,2,6,tzinfo=dt.timezone.utc)
RESERVE = 400 * 1024**2
EXPECTED = {"geometry__counterfact_true_false.pt":31964,
            "popqa__popqa_test_filtered_10shot.pt":6974,
            "x-fact__test_filtered_10shot.pt":2315}
TARGET_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
TOKENIZER_REVISION = "247cb7c3baebabc46cff1fb97a3a5913e4c4511d"
MAX_RECEIPT = 2 * 1024**2


class Cancelled(Exception):
    pass


class SpaceError(Exception):
    pass


class CommandError(Exception):
    pass


class CommandTimeout(Exception):
    pass


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def hash_file(path, guard=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8*1024**2):
            if guard:
                guard.check()
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".part")
    if temporary.is_symlink() or path.is_symlink():
        raise ValueError("Refusing a symlinked backup receipt")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def read_json(path):
    if path.is_symlink():
        raise ValueError("Refusing a symlinked input receipt")
    if not path.exists():
        return None
    if path.stat().st_size > MAX_RECEIPT:
        raise ValueError("Receipt exceeds the 2 MiB bound")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # A regular 120-second backup can be in progress; wait for its next copy.
        return None


def digest_value(value):
    return isinstance(value,str) and re.fullmatch(r"[a-f0-9]{64}",value) is not None


def positive_integer(value):
    return isinstance(value,int) and not isinstance(value,bool) and value>0


def canonical_source(value):
    if not isinstance(value,str) or ".." in PurePosixPath(value).parts:
        raise ValueError("Invalid source cache path")
    path = PurePosixPath(value)
    if not path.is_absolute():
        path = PurePosixPath(REMOTE_ROOT)/path
    allowed = {PurePosixPath("/dev/shm/mod1-transfer-cache"),
               PurePosixPath(REMOTE_ROOT+"/data/restored/mod1-transfer-cache")}
    if path.parent not in allowed or path.name not in EXPECTED:
        raise ValueError("Source is outside the three permitted target caches")
    return str(path)


def complete(receipt):
    """Read-only completion preconditions; an unfinished receipt is not ready."""
    if not receipt or receipt.get("status")!="complete_three_missing_caches":
        return False
    target,tokenizer=receipt.get("target",{}),receipt.get("tokenizer",{})
    parity=receipt.get("tokenizer_reference_parity",{})
    if not (target.get("model")=="unsloth/Llama-3.2-1B-Instruct" and target.get("revision")==TARGET_REVISION
            and target.get("all_files_match_pinned_historical_hashes") is True
            and tokenizer.get("model")=="VityaVitalich/Llama3.1-8b-instruct" and tokenizer.get("revision")==TOKENIZER_REVISION
            and tokenizer.get("all_files_match_pinned_historical_hashes") is True
            and parity.get("exact_chat_input_ids_match") is True and parity.get("exact_raw_claim_input_ids_match") is True
            and receipt.get("all_six_historical_caches_unchanged") is True):
        raise ValueError("Completed transfer receipt lacks pinned model/tokenizer or preservation proof")
    rows=receipt.get("new_caches",[])
    if len(rows)!=3 or {PurePosixPath(r.get("path","")).name for r in rows}!=set(EXPECTED):
        raise ValueError("Completion receipt must contain exactly the three new caches")
    for row in rows:
        path=canonical_source(row.get("path"))
        if row.get("sample_count")!=EXPECTED[PurePosixPath(path).name] or not positive_integer(row.get("raw_bytes")) or not digest_value(row.get("raw_sha256")):
            raise ValueError("Invalid source-cache count, size or SHA256")
        if not all(row.get(key) is True for key in ["claims_labels_match_csv_in_order","all_topk_widths_512","all_topk_logprob_length_offset_1"]):
            raise ValueError("Completed cache lacks row/alignment verification")
        if any(row.get("meta",{}).get(key)!=value for key,value in dict(hidden_size=2048,num_layers=16,num_attention_heads=32,vocab_size=128256,modalities="logits",model_dtype="torch.float32").items()):
            raise ValueError("Completed cache has unexpected 1B architecture or forward dtype")
    return True


def archive_complete(receipt, completion, completion_sha):
    """Validate complete archive metadata without writing or opening any cache."""
    if not receipt or receipt.get("status")!="complete_three_new_cache_archives":
        return False
    if not complete(completion) or receipt.get("source_receipt_sha256")!=completion_sha:
        raise ValueError("Archive receipt belongs to a different transfer completion")
    if receipt.get("original_raw_files_removed") is not False:
        raise ValueError("Archive receipt must explicitly preserve original raw files")
    rows=receipt.get("archives",[])
    sources={canonical_source(r["path"]):r for r in completion["new_caches"]}
    if len(rows)!=3 or {canonical_source(r.get("source")) for r in rows}!=set(sources):
        raise ValueError("Archive receipt must preserve exactly the three new source caches")
    for row in rows:
        source=canonical_source(row["source"])
        expected=sources[source]
        destination=PurePosixPath(row.get("archive",""))
        if ".." in destination.parts or destination!=PurePosixPath(REMOTE_ARCHIVES)/(PurePosixPath(source).name+".xz"):
            raise ValueError("Archive path is outside the fixed archive directory")
        if row.get("raw_sha256")!=expected["raw_sha256"] or row.get("raw_bytes")!=expected["raw_bytes"]:
            raise ValueError("Archive raw identity differs from the transfer completion")
        if not positive_integer(row.get("archive_bytes")) or not digest_value(row.get("archive_sha256")):
            raise ValueError("Invalid archive size or SHA256")
        if row.get("source_unchanged") is not True or row.get("roundtrip_raw_sha256_verified") is not True:
            raise ValueError("Archive lacks verified source preservation and lossless round trip")
    if receipt.get("total_archive_bytes")!=sum(r["archive_bytes"] for r in rows):
        raise ValueError("Archive total byte count differs")
    return True


class Guard:
    def __init__(self,deadline=HARD_STOP,cancel_paths=(),stop=None,now=utcnow):
        self.deadline,self.cancel_paths,self.stop,self.now=deadline,list(cancel_paths),stop or threading.Event(),now

    def check(self):
        if self.stop.is_set() or any(path.exists() for path in self.cancel_paths):
            raise Cancelled("Local manual stop or watchdog.cancel requested")
        if self.now()>=self.deadline:
            raise TimeoutError("Local cache-backup hard stop reached")


def terminate(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def monitored(command,guard,timeout=45,space_path=None,reserve=RESERVE):
    """Bound subprocesses, including downloads, by cancellation and UTC deadline."""
    guard.check()
    started=time.monotonic()
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try:
        while True:
            guard.check()
            if time.monotonic()-started>=timeout:
                raise CommandTimeout("Bounded SSH/transfer command timed out")
            if space_path is not None and shutil.disk_usage(space_path).free<reserve:
                raise SpaceError("Local free space dropped below the protected reserve")
            try:
                stdout,stderr=child.communicate(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                pass
        if child.returncode:
            raise CommandError(f"SSH/rsync exit {child.returncode}: {stderr[-1500:]}")
        return stdout
    finally:
        terminate(child)
        for stream in [child.stdout,child.stderr]:
            stream.close()


def ssh_command(code,*arguments):
    return SSH+[HOST,shlex.join(["python3","-c",code,*arguments])]


# This fixed remote transaction detaches only the existing CPU archiver. The
# inherited flock remains held by the archiver for its entire lifetime.
REMOTE_LAUNCH_CODE = r'''
import datetime as dt,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT = Path('/workspace/mod1')
out=ROOT/'results/overnight20261021'
completion=out/'remaining_transfer_completion.json'
manifest=out/'new_feature_cache_archives.json'
script=ROOT/'scripts/overnight_archive_transfer.py'
cutoff=dt.datetime(2026,10,22,2,6,tzinfo=dt.timezone.utc)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(status,**values):print(json.dumps(dict(status=status,**values)),flush=True)
if dt.datetime.now(dt.timezone.utc)>=cutoff:
    emit('archive_cutoff_reached');sys.exit(0)
if completion.is_symlink() or script.is_symlink() or manifest.is_symlink():
    emit('unsafe_remote_source_or_receipt_path');sys.exit(0)
if not completion.is_file() or sha(completion)!=sys.argv[1]:
    emit('waiting_for_matching_completion_receipt');sys.exit(0)
if not script.is_file() or sha(script)!=sys.argv[2]:
    emit('archive_script_sha_mismatch');sys.exit(0)
receipt=json.loads(completion.read_text())
if receipt.get('status')!='complete_three_missing_caches':
    emit('waiting_for_completed_transfer_receipt');sys.exit(0)
out.mkdir(parents=True,exist_ok=True)
if (out/'cache_backup.archive.lock').is_symlink():
    emit('unsafe_remote_lock_path');sys.exit(0)
lock=open(out/'cache_backup.archive.lock','a')
try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    emit('archive_lock_busy');sys.exit(0)
if manifest.exists():
    old=json.loads(manifest.read_text())
    if old.get('source_receipt_sha256')!=sys.argv[1]:
        emit('archive_receipt_identity_mismatch');sys.exit(0)
    if old.get('status')=='complete_three_new_cache_archives':
        emit('archive_already_complete');sys.exit(0)
    if old.get('status')=='partial_or_failed':
        emit('archive_failed_requires_review');sys.exit(0)
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:argv=(proc/'cmdline').read_bytes().split(b'\0')
    except (OSError,PermissionError):continue
    if b'--execute' in argv and any(a in [str(script).encode(),b'scripts/overnight_archive_transfer.py'] for a in argv):
        emit('existing_archive_process',pid=int(proc.name));sys.exit(0)
if manifest.exists():
    emit('stale_archive_receipt_requires_review');sys.exit(0)
if (out/'cache_archive_backup_launch.json').exists():
    emit('archive_launch_died_requires_review');sys.exit(0)
logpath=out/'cache_archive_backup.remote.log'
command=[str(ROOT/'.venv/bin/python'),str(script),'--execute','--workers','2','--deadline-utc','2026-10-22T02:06:00Z']
with logpath.open('ab') as log:
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
state=dict(status='launched_detached_cpu_archiver',pid=child.pid,log=str(logpath),command=command,lock=str(out/'cache_backup.archive.lock'),
           source_receipt_sha256=sys.argv[1],archive_script_sha256=sys.argv[2],started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),archive_cutoff_utc=cutoff.isoformat())
target=out/'cache_archive_backup_launch.json';temp=target.with_suffix('.json.part')
temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(target)
emit(**state)
'''

REMOTE_READ_CODE = r'''
import json
from pathlib import Path
p=Path('/workspace/mod1/results/overnight20261021/new_feature_cache_archives.json')
if not p.exists():print(json.dumps(dict(status='waiting_for_archive_receipt')))
elif p.is_symlink() or p.stat().st_size>2097152:raise ValueError('Unsafe archive receipt')
else:print(p.read_text())
'''


def local_paths(directory,row):
    name=PurePosixPath(row["archive"]).name
    if name not in {n+".xz" for n in EXPECTED}:
        raise ValueError("Unexpected local archive name")
    if directory.is_symlink() or directory.resolve()!=directory.absolute():
        raise ValueError("Local archive directory contains a symlink")
    target=directory/name
    partial=directory/(name+".download.part")
    if target.is_symlink() or partial.is_symlink():
        raise ValueError("Refusing a symlinked final or partial archive")
    return target,partial


def local_verified(path,row,guard):
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size!=row["archive_bytes"] or hash_file(path,guard)!=row["archive_sha256"]:
        raise ValueError("Existing local final archive conflicts; refusing overwrite: "+path.name)
    return True


def space_plan(rows,directory,guard,reserve=RESERVE,free=None):
    guard.check()
    remaining,verified=0,[]
    for row in rows:
        target,partial=local_paths(directory,row)
        if local_verified(target,row,guard):
            verified.append(target.name)
            continue
        allocated=0
        if partial.exists():
            if not partial.is_file() or partial.stat().st_size>row["archive_bytes"]:
                raise ValueError("Invalid or oversized partial archive; manual review required")
            stat=partial.stat()
            # Sparse logical bytes are not already-paid-for disk allocation.
            allocated=min(stat.st_size,getattr(stat,"st_blocks",0)*512)
        remaining+=max(0,row["archive_bytes"]-allocated)
    if free is None:
        probe=directory if directory.exists() else directory.parent
        free=shutil.disk_usage(probe).free
    return dict(remaining_archive_bytes=remaining,available_bytes=free,protected_reserve_bytes=reserve,
                required_free_bytes=remaining+reserve,space_admitted=free>=remaining+reserve,already_verified=verified)


def fetch_archives(receipt,directory,guard,record,runner=monitored,reserve=RESERVE):
    guard.check()
    if directory.is_symlink() or directory.resolve()!=directory.absolute():
        raise ValueError("Local archive directory contains a symlink")
    directory.mkdir(parents=True,exist_ok=True)
    rows=sorted(receipt["archives"],key=lambda r:(-r["archive_bytes"],r["archive"]))
    plan=space_plan(rows,directory,guard,reserve)
    if not plan["space_admitted"]:
        record("insufficient_local_space",**plan)
        raise SpaceError("All three remaining archives plus reserve do not fit; no download started")
    record("space_admitted",**plan)
    verified=[]
    for row in rows:
        guard.check()
        target,partial=local_paths(directory,row)
        if not local_verified(target,row,guard):
            plan=space_plan(rows,directory,guard,reserve)
            if not plan["space_admitted"]:
                record("insufficient_local_space",**plan)
                raise SpaceError("Free space changed before the next archive")
            command=["rsync","-t","--partial","--inplace","--checksum","--protect-args","--timeout=30",
                     "-e",shlex.join(SSH),HOST+":"+row["archive"],str(partial)]
            record("downloading_archive",archive=target.name,expected_bytes=row["archive_bytes"])
            runner(command,guard,timeout=max(1,(guard.deadline-guard.now()).total_seconds()),space_path=directory,reserve=reserve)
            guard.check()
            if partial.stat().st_size!=row["archive_bytes"] or hash_file(partial,guard)!=row["archive_sha256"]:
                raise ValueError("Downloaded archive size/SHA256 mismatch; partial retained for review")
            if target.exists():
                raise FileExistsError("A final archive appeared during download; refusing replacement")
            # A hard link publishes without ever replacing a concurrent final.
            # Remove only this helper's private, fully verified download name.
            os.link(partial,target)
            partial.unlink()
        entry=dict(file=str(target),archive_sha256=row["archive_sha256"],archive_bytes=row["archive_bytes"],
                   raw_sha256=row["raw_sha256"],raw_bytes=row["raw_bytes"],compressed_sha256_verified=True,
                   remote_roundtrip_raw_sha256_verified=True)
        verified.append(entry)
        record("archive_verified",files=verified.copy(),archive=target.name)
    return verified


class Watcher:
    def __init__(self,guard,output=OUTPUT,destination=DESTINATION,runner=monitored,reserve=RESERVE):
        self.guard,self.output,self.destination,self.runner,self.reserve=guard,output,destination,runner,reserve
        self.receipt_path=output/"cache_backup_receipt.json"
        self.state=read_json(self.receipt_path) or dict(files=[])

    def record(self,status,**values):
        self.state.update(status=status,updated_utc=utcnow().isoformat(),hard_stop_utc=self.guard.deadline.isoformat(),
                          no_gpu_or_feature_extraction=True,no_pod_stop=True,**values)
        atomic_json(self.receipt_path,self.state)
        with (self.output/"cache_backup.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(status=status,utc=utcnow().isoformat(),**values),allow_nan=False)+"\n")

    def cycle(self):
        self.guard.check()
        completion_path=self.output/"remaining_transfer_completion.json"
        completion=read_json(completion_path)
        if not complete(completion):
            self.record("waiting_for_completed_transfer_receipt")
            return False
        completion_sha=hash_file(completion_path,self.guard)
        self.record("transfer_completion_verified",source_completion_sha256=completion_sha)
        # Reuse a valid, already backed-up manifest without any network request.
        archive=read_json(self.output/"new_feature_cache_archives.json")
        if not archive_complete(archive,completion,completion_sha):
            archive=read_json(self.output/"cache_backup_remote_archive_receipt.json")
        if not archive_complete(archive,completion,completion_sha):
            raw=self.runner(ssh_command(REMOTE_READ_CODE),self.guard,timeout=45)
            if len(raw.encode())>MAX_RECEIPT:
                raise ValueError("Remote archive receipt exceeds bound")
            archive=json.loads(raw)
            atomic_json(self.output/"cache_backup_remote_archive_receipt.json",archive)
        if not archive_complete(archive,completion,completion_sha):
            if archive.get("status")=="partial_or_failed":
                self.record("remote_archive_failed_requires_review",remote_error=str(archive.get("error",""))[:1500])
                return False
            if self.guard.now()<ARCHIVE_CUTOFF:
                self.guard.check()
                source_script=ROOT/"scripts/overnight_archive_transfer.py"
                response=self.runner(ssh_command(REMOTE_LAUNCH_CODE,completion_sha,hash_file(source_script,self.guard)),self.guard,timeout=45)
                launched=json.loads(response)
                self.record("remote_archive_status",remote_launch=launched)
                if launched["status"] in {"archive_script_sha_mismatch","archive_receipt_identity_mismatch","archive_failed_requires_review","stale_archive_receipt_requires_review","archive_launch_died_requires_review","unsafe_remote_source_or_receipt_path","unsafe_remote_lock_path"}:
                    raise ValueError("Remote archiver launch precondition failed: "+launched["status"])
            else:
                self.record("archive_cutoff_passed_waiting_for_receipt",archive_status=archive.get("status"))
            return False
        atomic_json(self.output/"cache_backup_remote_archive_receipt.json",archive)
        self.record("all_three_remote_archives_verified",remote_archive_receipt_sha256=hashlib.sha256(json.dumps(archive,sort_keys=True).encode()).hexdigest())
        files=fetch_archives(archive,self.destination,self.guard,self.record,self.runner,self.reserve)
        self.record("complete_three_local_cache_archives",files=files,total_archive_bytes=sum(r["archive_bytes"] for r in files))
        return True


def parse_deadline(value):
    result=dt.datetime.fromisoformat(value.replace("Z","+00:00"))
    if result.tzinfo is None or result.utcoffset()!=dt.timedelta(0) or result>HARD_STOP:
        raise argparse.ArgumentTypeError("Use UTC no later than 2026-10-22T02:13:00Z")
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--once",action="store_true",help="One watcher cycle, without waiting for the next backup")
    ap.add_argument("--poll-seconds",type=float,default=30)
    ap.add_argument("--reserve-mib",type=int,default=400)
    ap.add_argument("--hard-stop-utc",type=parse_deadline,default=HARD_STOP)
    args=ap.parse_args()
    if not 1<=args.poll_seconds<=120 or args.reserve_mib<400:
        ap.error("Poll interval must be 1..120s and disk reserve at least 400 MiB")
    cancel_paths=[OUTPUT/name for name in ["watchdog.cancel","cache_backup.cancel","manual.stop","manual_stop"]]
    guard=Guard(args.hard_stop_utc,cancel_paths)
    if not args.execute:
        completion=read_json(OUTPUT/"remaining_transfer_completion.json")
        print(json.dumps(dict(status="dry_run_no_ssh_or_launch",completed_transfer_ready=complete(completion),
            expected_new_caches=list(EXPECTED),local_destination=str(DESTINATION),reserve_bytes=args.reserve_mib*1024**2,
            archive_cutoff_utc=ARCHIVE_CUTOFF.isoformat(),hard_stop_utc=args.hard_stop_utc.isoformat(),
            cancellation_files=[str(p) for p in cancel_paths],remote_cpu_archive_only=True,execute=False)))
        return
    if ROOT.resolve()!=Path("/root/evohallu/mod1").resolve():
        ap.error("This watcher may execute only from the local /root/evohallu/mod1 checkout, never on the pod")
    if not Path("/root/.ssh/processbench_pod_ed25519").is_file():
        ap.error("The configured local SSH identity is unavailable")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock_path=OUTPUT/"cache_backup.local.lock"
    if lock_path.is_symlink():
        raise ValueError("Refusing a symlinked local backup lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status="another_local_backup_watcher_is_running")))
            return
        watcher=Watcher(guard,reserve=args.reserve_mib*1024**2)
        old={sig:signal.getsignal(sig) for sig in [signal.SIGTERM,signal.SIGINT]}
        for sig in old:
            signal.signal(sig,lambda signum,frame:guard.stop.set())
        try:
            while True:
                try:
                    if watcher.cycle() or args.once:
                        break
                except SpaceError as exc:
                    watcher.record("insufficient_local_space",error=str(exc))
                    if args.once:
                        break
                except (CommandError,CommandTimeout) as exc:
                    watcher.record("transient_ssh_or_transfer_error_will_retry",error=str(exc))
                    if args.once:
                        break
                until=time.monotonic()+args.poll_seconds
                while time.monotonic()<until:
                    guard.check()
                    time.sleep(min(.25,max(0,until-time.monotonic())))
        except Cancelled as exc:
            watcher.record("cancelled_after_manual_stop",error=str(exc))
        except TimeoutError as exc:
            watcher.record("stopped_at_bounded_deadline_or_timeout",error=str(exc))
        except Exception as exc:
            watcher.record("failed_requires_review",error=type(exc).__name__+": "+str(exc))
            raise
        finally:
            for sig,handler in old.items():
                signal.signal(sig,handler)
        print(json.dumps(dict(status=watcher.state["status"],receipt=str(watcher.receipt_path))))


if __name__=="__main__":
    main()
