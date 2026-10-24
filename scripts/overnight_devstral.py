"""Optional native Devstral writer pair; dry-run unless explicitly executed.

--verify-prefetch verifies an already downloaded native checkpoint and writes its
ready receipt. --execute owns two idle-GPU servers and one fixed-budget pair.
Neither mode installs packages, rotates checkpoints or changes pinned core code.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from omegaconf import OmegaConf

try:
    from scripts import overnight_prepare_confirmation as discovery
    from scripts import overnight_start_glm as hardware
    from scripts import overnight_followups as common
    from scripts import run_overnight_parallel as controller
except ModuleNotFoundError:
    import overnight_prepare_confirmation as discovery
    import overnight_start_glm as hardware
    import overnight_followups as common
    import run_overnight_parallel as controller

ROOT = Path(__file__).resolve().parents[1]
MODEL = WRITER = "mistralai/Devstral-Small-2-24B-Instruct-2512"
REVISION = "55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128"
MODEL_PATH = Path("/dev/shm/overnight-models/Devstral-Small-2-24B")
OUT = ROOT / "results/overnight20261021/devstral"
SERVER_METADATA = ROOT / "out/overnight20261021/devstral_servers.json"
CUTOFF = dt.datetime(2026, 10, 22, 1, 10, tzinfo=dt.timezone.utc)
WEIGHT_BYTES = 25793035888
NATIVE_SHARDS = {"consolidated-00001-of-00002.safetensors", "consolidated-00002-of-00002.safetensors"}
PRECISION = "published FP8 weights / BF16 activations / A100 Marlin weight-only FP8; no native FP8 compute"
PORTS = (18000, 18001)
GPUS = (0, 1)


def now():
    return dt.datetime.now(dt.timezone.utc)


def sha(path):
    return controller.sha(path)


def read(path):
    return json.loads(path.read_text())


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()
    if path.exists():
        raise FileExistsError("Refusing to overwrite frozen optional artifacts: "+str(path))
    with path.open("xb") as stream:
        stream.write(encoded)


def parse_cutoff(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0) or parsed > CUTOFF:
        raise ValueError("Devstral deadline must be aware UTC and cannot exceed 2026-10-22T01:10:00Z")
    return parsed


def time_gate(cutoff, minimum_minutes, at=None):
    remaining = (cutoff-(at or now())).total_seconds()/60
    if remaining < minimum_minutes:
        raise TimeoutError(f"Need {minimum_minutes} minutes before the writer cutoff; only {remaining:.2f} remain")
    return remaining


def verify_native(path, root=ROOT):
    """Rehash all selected files against public LFS SHA256 or Git blob SHA1."""
    metadata_path, manifest_path = path/"download_metadata.json", path/"download_manifest.json"
    metadata, manifest = read(metadata_path), read(manifest_path)
    if (metadata.get("id") != MODEL or metadata.get("sha") != REVISION or
            metadata.get("private") or metadata.get("gated") or
            manifest.get("model") != MODEL or manifest.get("revision") != REVISION):
        raise ValueError("Expected the public, ungated, pinned official Devstral checkpoint")
    published = {r["rfilename"]:r for r in metadata["siblings"]}
    files = manifest["files"]
    names = {r["file"] for r in files}
    if len(names) != len(files) or {name for name in names if name.endswith(".safetensors")} != NATIVE_SHARDS:
        raise ValueError("Native-only download must contain the exact two consolidated shards")
    if any(p.name.startswith("model-") for p in path.glob("*.safetensors")):
        raise ValueError("Refusing duplicate HF and native checkpoint formats")
    index = read(path/"consolidated.safetensors.index.json")
    if set(index["weight_map"].values()) != NATIVE_SHARDS:
        raise ValueError("Native shard index differs from pinned files")
    required = {"params.json","config.json","tekken.json","tokenizer_config.json","chat_template.jinja",
                "consolidated.safetensors.index.json"}
    if not required <= names:
        raise ValueError("Missing native model/tokenizer configuration")
    rows=[]
    for receipt in files:
        name=receipt["file"]
        if Path(name).name != name or name not in published:
            raise ValueError("Unexpected native checkpoint file")
        file=path/name; public=published[name]
        if file.stat().st_size != public["size"] or receipt["bytes"] != public["size"]:
            raise ValueError("Checkpoint file size mismatch: "+name)
        digest=sha(file)
        lfs=public.get("lfs",{}).get("sha256")
        if lfs:
            if digest != lfs:
                raise ValueError("Published LFS SHA256 mismatch: "+name)
            proof=dict(published_sha256=lfs,verification="published_lfs_sha256")
        else:
            data=file.read_bytes()
            blob=hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
            if blob != public.get("blobId"):
                raise ValueError("Published Git blob SHA1 mismatch: "+name)
            proof=dict(published_git_blob_sha1=blob,verification="published_git_blob_sha1_plus_recorded_sha256")
        rows.append(dict(file=name,bytes=file.stat().st_size,sha256=digest,**proof))
    if sum(r["bytes"] for r in rows if r["file"].endswith(".safetensors")) != WEIGHT_BYTES:
        raise ValueError("Native weight bytes differ from the pinned checkpoint")
    config=read(path/"config.json")
    if config.get("architectures") != ["Mistral3ForConditionalGeneration"] or config.get("dtype") != "bfloat16" or config.get("quantization_config",{}).get("quant_method") != "fp8":
        raise ValueError("Unexpected Devstral architecture/dtype/quantization")
    model=OmegaConf.load(root/"configs/model/devstral_small2.yaml")
    if model.hf_model_id != MODEL or model.revision != REVISION or model.supports_chat_template_kwargs is not False:
        raise ValueError("Inherited Devstral writer config differs from the native tokenizer contract")
    for key,value in dict(temperature=.7,topk=20,topp=.8,max_tokens=4096,repetition_penalty=1.).items():
        if model[key] != value:raise ValueError("Unexpected Devstral decoder setting: "+key)
    return dict(status="ready_native_checkpoint",model=MODEL,revision=REVISION,path=str(path.resolve()),
        native_files=rows,weight_bytes=WEIGHT_BYTES,total_file_bytes=sum(r["bytes"] for r in rows),precision=PRECISION,
        download_manifest_sha256=sha(manifest_path),download_metadata_sha256=sha(metadata_path))


def prefetch_receipt(path):
    proof=verify_native(path)
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=OUT/"prefetch_receipt.json"
    if receipt.exists():
        previous=read(receipt)
        if any(previous.get(k)!=v for k,v in proof.items()):
            raise ValueError("Existing prefetch receipt differs from verified native files")
        return previous
    copies=OUT/"model_provenance";copies.mkdir(exist_ok=True)
    for name in ["download_manifest.json","download_metadata.json","config.json","params.json",
                 "tokenizer_config.json","chat_template.jinja","generation_config.json","consolidated.safetensors.index.json"]:
        shutil.copyfile(path/name,copies/name)
    proof["verified_utc"]=now().isoformat()
    controller.atomic_json(receipt,proof)
    return proof


def config_bytes(policy):
    config=dict(defaults=["/config_uq_trial",{"override /model":"devstral_small2"},"_self_"],prefix="research20261021",
        run_identifier_name="nextdevstral_evo_"+policy,seed=2107,search_mode="evolution",num_rounds=30,num_outputs_per_prompt=2,
        model=dict(max_tokens=4096),trial=dict(prompt_policy=policy,save_train_predictions=True,shortlist_size=3,evaluation_workers=1,
        max_seconds=3600,max_total_tokens=2000000,max_consecutive_failed_rounds=31))
    return ("# @package _global_\n"+OmegaConf.to_yaml(OmegaConf.create(config))).encode()


def prepare(source_path, receipt_path, root=ROOT):
    source=read(source_path)
    selected=discovery.inspect_discovery(source,root=root)
    if not selected["ready"] or selected["selected_policy"] != "diverse":
        raise ValueError("The fixed diverse comparison must be justified by all twelve full source-validation runs")
    jobs=[]
    for policy in ("original","diverse"):
        name=f"nextdevstral_evo_{policy}_2107";config="next_experiments/"+name
        path=root/"configs"/(config+".yaml");data=config_bytes(policy)
        if path.exists():
            raise FileExistsError("Refusing to repeat/replace a Devstral arm")
        jobs.append(dict(id=name,phase="evolution_screen",study_phase="writer_generalization",mode="evolution",
            prompt_policy=policy,seed=2107,requests=60,config=config,config_sha256=hashlib.sha256(data).hexdigest(),
            command=[".venv/bin/python","src/experiments/main.py","--config-name",config],
            expected_logs="out/logs/research20261021/uncertainty_quantification_"+name))
    pins=dict(source["study_source_sha256"])
    for file in ["scripts/overnight_devstral.py","scripts/overnight_prepare_confirmation.py","configs/model/devstral_small2.yaml"]:
        pins[file]=sha(root/file)
    manifest=dict(created_utc=now().isoformat(),status="prepared_optional_writer_pair",study_phase="writer_generalization",
        writer=MODEL,writer_revision=REVISION,feature_model=source["feature_model"],labels="is_hallu",seeds=[2107],jobs=jobs,
        training_raw_path=source["training_raw_path"],training_raw_sha256=source["training_raw_sha256"],
        training_csv_sha256=source["training_csv_sha256"],study_source_sha256=pins,requests_by_phase={"evolution_screen":120},
        selection_provenance=dict(source_manifest=str(source_path.relative_to(root)),source_manifest_sha256=sha(source_path),**selected),
        prefetch_receipt=str(receipt_path.relative_to(root)),prefetch_receipt_sha256=sha(receipt_path),precision=PRECISION,
        decoding=dict(temperature=.7,top_p=.8,top_k=20,max_tokens=4096,enable_thinking=False,native_tokenizer=True),
        optional_schedule=dict(writer_cutoff_utc=CUTOFF.isoformat(),entry_minimum_minutes=75,ready_minimum_minutes=65,
            maximum_startup_minutes=10,max_job_minutes=60,min_job_minutes=60,no_repair_or_retry=True))
    target=root/"results/next_experiments/devstral_seed2107_manifest.json"
    if target.exists():
        raise FileExistsError("Refusing to repeat the optional Devstral execution manifest")
    # Check existing pins before writing any trial files; verify_inputs checks
    # the generated configs and newly pinned helper before model serving.
    for file,pin in source["study_source_sha256"].items():
        if sha(root/file) != pin:
            raise ValueError("Pinned source changed: "+file)
    for job in jobs:
        path=root/"configs"/(job["config"]+".yaml");path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(config_bytes(job["prompt_policy"]))
    exclusive(target,manifest)
    controller.verify_inputs(manifest,target,jobs,root=root)
    return target,manifest


def merge_analysis(path,root=ROOT):
    manifest=read(path);target=root/"results/overnight20261021/analysis_manifest.json"
    if manifest["writer"] != MODEL or len(manifest["jobs"]) != 2:
        raise ValueError("Unexpected Devstral pair")
    with target.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        analysis=read(target);jobs={r["id"]:r for r in analysis["jobs"]}
        provenance={r["path"]:r for r in analysis.get("execution_manifests",[])}
        relative=str(path.relative_to(root));digest=sha(path)
        if relative in provenance and provenance[relative]["sha256"] != digest:
            raise ValueError("Existing Devstral execution manifest changed")
        for original in manifest["jobs"]:
            row=copy.deepcopy(original);row.update(phase="writer_generalization",execution_phase=original["phase"],
                execution_manifest=relative,execution_manifest_sha256=digest,writer_model=MODEL)
            if row["id"] in jobs and jobs[row["id"]] != row:
                raise ValueError("Existing Devstral analysis identity changed")
            jobs[row["id"]]=row
        provenance[relative]=dict(path=relative,sha256=digest)
        totals={}
        for row in jobs.values():totals[row["phase"]]=totals.get(row["phase"],0)+row["requests"]
        analysis.update(jobs=list(jobs.values()),execution_manifests=list(provenance.values()),requests_by_phase=totals,updated_utc=now().isoformat())
        controller.atomic_json(target,analysis)


def server_command(path,port):
    return [str(ROOT/".venv/bin/vllm"),"serve",str(path),"--served-model-name",MODEL,"--host","127.0.0.1","--port",str(port),
        "--dtype","bfloat16","--tokenizer-mode","mistral","--config-format","mistral","--load-format","mistral",
        "--language-model-only","--linear-backend","marlin","--max-model-len","16384","--max-num-seqs","2",
        "--gpu-memory-utilization","0.90","--tensor-parallel-size","1","--enforce-eager"]


def controller_command(path,state,cutoff):
    # The unchanged controller reserves thirty minutes from its deadline.
    return [sys.executable,"scripts/run_overnight_parallel.py","--execute","--manifest",str(path),"--phase","all",
        "--state",str(state),"--ports",*map(str,PORTS),"--max-job-minutes","60","--min-job-minutes","60",
        "--deadline-utc",(cutoff+dt.timedelta(minutes=30)).isoformat(),"--reserve-minutes","30"]


def ready(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",timeout=3) as stream:
            return MODEL in {row["id"] for row in json.load(stream).get("data",[])}
    except (OSError,ValueError):return False


def stop_owned(children):
    for child in reversed(children):
        # A directly owned, unreaped Popen PID cannot have been reused.
        if child.poll() is None:
            if os.getpgid(child.pid) != child.pid:
                raise RuntimeError("Owned child lost its expected process group")
            hardware.terminate(child)


def cleanup_trial_groups(state_path,manifest_path,manifest):
    """Fallback for trial sessions if their controller needed forced termination."""
    if not state_path.exists() or manifest is None:return
    state=read(state_path)
    if state.get("manifest_sha256") != sha(manifest_path):
        raise ValueError("Refusing cleanup from an unrelated controller state")
    lookup={job["id"]:job for job in manifest["jobs"]}
    def validate(record):
        pid=record["pid"]
        if not isinstance(pid,int) or isinstance(pid,bool) or pid<=1 or record["job"] not in lookup or record["port"] not in PORTS:
            raise ValueError("Unrecognized owned trial process record")
        if not common.process_alive(pid):return False
        saved=state["jobs"][record["job"]]
        expected=controller.child_command(lookup[record["job"]],record["port"],saved["max_seconds"])
        if saved["command"] != expected:raise ValueError("Changed owned trial command")
        actual=[s.decode() for s in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if s]
        if actual[-(len(expected)-1):] != expected[1:] or Path(f"/proc/{pid}/cwd").resolve()!=ROOT.resolve() or os.getpgid(pid)!=pid:
            raise ValueError("PID no longer identifies the owned trial process group")
        return True
    records=state.get("running",[])
    for record in records:
        if validate(record):os.killpg(record["pid"],signal.SIGTERM)
    until=time.monotonic()+5
    while any(common.process_alive(r["pid"]) for r in records) and time.monotonic()<until:time.sleep(.1)
    for record in records:
        if validate(record):os.killpg(record["pid"],signal.SIGKILL)


def execute(args):
    cutoff=parse_cutoff(args.deadline_utc)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"orchestrator.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/"decision.json").exists():raise FileExistsError("Devstral has already been decided; no retry")
        children=[];metadata=None;manifest=None;path=None;state=ROOT/"results/next_experiments/devstral_execution_state.json"
        decision=dict(started_utc=now().isoformat(),model=MODEL,revision=REVISION,precision=PRECISION,status="not_started",writer_cutoff_utc=cutoff.isoformat())
        def interrupted(signum,frame):raise KeyboardInterrupt(f"Devstral received signal {signum}")
        def expired(signum,frame):raise TimeoutError("Devstral hard writer cutoff reached")
        oldterm=signal.signal(signal.SIGTERM,interrupted);oldalarm=signal.signal(signal.SIGALRM,expired)
        try:
            time_gate(cutoff,75)
            if not (OUT/"prefetch_receipt.json").exists():raise FileNotFoundError("Native prefetch receipt is not ready")
            proof=prefetch_receipt(args.model_path)
            if importlib.metadata.version("vllm").split("+")[0] != "0.31.0":raise ValueError("Expected audited vLLM 0.31.0")
            if SERVER_METADATA.exists():
                if any(common.process_alive(r["pid"]) for r in read(SERVER_METADATA).get("servers",[])):
                    raise RuntimeError("Existing managed Devstral PIDs require inspection")
            hardware.require_idle(hardware.telemetry(),GPUS);hardware.require_free_ports(PORTS)
            path,manifest=prepare(args.source_manifest,OUT/"prefetch_receipt.json")
            time_gate(cutoff,75)
            signal.setitimer(signal.ITIMER_REAL,max(.001,(cutoff-now()).total_seconds()))
            metadata=dict(model=MODEL,revision=REVISION,precision=PRECISION,started_utc=now().isoformat(),status="starting",servers=[],
                checkpoint_receipt_sha256=sha(OUT/"prefetch_receipt.json"),helper_sha256=sha(Path(__file__)),
                runtime_versions={name:importlib.metadata.version(name) for name in ("vllm","transformers","mistral-common")},
                native_tokenizer=True,language_model_only=True,eager=True,linear_backend="marlin")
            SERVER_METADATA.parent.mkdir(parents=True,exist_ok=True)
            until=min(cutoff.timestamp(),time.time()+600)
            for gpu,port in zip(GPUS,PORTS):
                if time.time()>=until:raise TimeoutError("Devstral startup deadline reached before replica launch")
                hardware.require_idle(hardware.telemetry(),[gpu]);hardware.require_free_ports([port])
                command=server_command(args.model_path,port);log=OUT/f"server_gpu{gpu}.log"
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",VLLM_NO_USAGE_STATS="1")
                with log.open("a") as stream:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child);metadata["servers"].append(dict(pid=child.pid,gpu=gpu,port=port,command=command,log=str(log)))
                controller.atomic_json(SERVER_METADATA,metadata)
            while time.time()<until:
                if any(child.poll() is not None for child in children):raise RuntimeError("Devstral replica exited during startup; no retry")
                if all(ready(port) for port in PORTS):break
                time.sleep(min(5,max(.1,until-time.time())))
            else:raise TimeoutError("Devstral replicas did not become ready within ten minutes")
            time_gate(cutoff,65)
            common.validated_servers(SERVER_METADATA,MODEL,REVISION)
            metadata.update(status="ready",ready_utc=now().isoformat());controller.atomic_json(SERVER_METADATA,metadata)
            merge_analysis(path)
            command=controller_command(path,state,cutoff)
            with (OUT/"controller.log").open("a") as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH="src:."),stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);decision.update(status="running",execution_manifest=str(path.relative_to(ROOT)),execution_manifest_sha256=sha(path),controller_pid=child.pid)
            code=child.wait(timeout=max(1,(cutoff-now()).total_seconds()))
            terminal=read(state)
            stopped=bool(terminal.get("events")) and terminal["events"][-1].get("event")=="controller_stopped" and not terminal.get("running")
            if terminal.get("manifest_sha256")!=sha(path) or not stopped:raise RuntimeError("Controller lacks pinned terminal evidence")
            complete=[job["id"] for job in manifest["jobs"] if controller.inspect_job(job)=="complete"]
            decision.update(status="completed_full_pair" if code==0 and len(complete)==2 else "failed_or_partial_pair",
                controller_exit_code=code,complete_jobs=complete,controller_terminal_proved=True)
        except BaseException as exc:
            decision.update(status="skipped_or_failed" if decision["status"]!="running" else "failed_or_partial_pair",error=f"{type(exc).__name__}: {exc}")
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            try:
                stop_owned(children)
                cleanup_trial_groups(state,path,manifest)
                decision["owned_process_cleanup"]=True
            except BaseException as exc:
                decision["owned_process_cleanup"]=False;decision["cleanup_error"]=repr(exc)
            if metadata is not None:
                metadata.update(status="stopped" if decision.get("owned_process_cleanup") else "cleanup_failed",
                    stopped_utc=now().isoformat(),owned_process_cleanup=decision.get("owned_process_cleanup"))
                controller.atomic_json(SERVER_METADATA,metadata)
            if decision.get("owned_process_cleanup") is False:decision["status"]="cleanup_failed"
            decision.update(finished_utc=now().isoformat(),helper_sha256=sha(Path(__file__)),pod_stopped=False)
            controller.atomic_json(OUT/"decision.json",decision)
            signal.signal(signal.SIGTERM,oldterm);signal.signal(signal.SIGALRM,oldalarm)
        return decision


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path",type=Path,default=MODEL_PATH)
    ap.add_argument("--source-manifest",type=Path,default=ROOT/"results/next_experiments/run_manifest.json")
    ap.add_argument("--deadline-utc",default="2026-10-22T01:10:00Z")
    action=ap.add_mutually_exclusive_group();action.add_argument("--execute",action="store_true");action.add_argument("--verify-prefetch",action="store_true")
    args=ap.parse_args();args.source_manifest=args.source_manifest.resolve();os.chdir(ROOT)
    cutoff=parse_cutoff(args.deadline_utc)
    if args.verify_prefetch:
        print(json.dumps(prefetch_receipt(args.model_path),indent=2));return
    if args.execute:
        result=execute(args);print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="completed_full_pair" else 2)
    selected=discovery.inspect_discovery(read(args.source_manifest))
    print(json.dumps(dict(dry_run=True,model=MODEL,revision=REVISION,precision=PRECISION,source_selection=selected,
        prefetch_receipt=str(OUT/"prefetch_receipt.json"),prefetch_ready=(OUT/"prefetch_receipt.json").exists(),
        entry_minutes=75,ready_minutes=65,startup_max_minutes=10,writer_cutoff_utc=cutoff.isoformat(),
        planned_arms=["original","diverse"],seed=2107,requests_per_arm=60,no_gpu_work=True),indent=2))


if __name__=="__main__":main()
