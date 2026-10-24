"""Start optional pinned GLM replicas only on idle GPUs and free local ports.

Default is a dry run. No downloads or inference requests are performed. --execute
validates checkpoint receipts and actual shard sizes, starts eager BF16 servers,
and records actual GPU telemetry while awaiting local readiness.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
WRITER = "zai-org/GLM-4.7-Flash"
REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
WEIGHT_BYTES = 62444175504
WEIGHT_SHARDS = 48
CUTOFF = dt.datetime.fromisoformat("2026-10-22T01:10:00+00:00")


def utc():
    return dt.datetime.now(dt.timezone.utc)


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def verify_checkpoint(path, verify_all_hashes=False):
    manifest_path, metadata_path = path / "download_manifest.json", path / "download_metadata.json"
    manifest, metadata = read(manifest_path), read(metadata_path)
    if manifest.get("model") != WRITER or manifest.get("revision") != REVISION or metadata.get("sha") != REVISION:
        raise ValueError("Checkpoint download provenance differs from the pinned GLM model/revision")
    if metadata.get("private") or metadata.get("gated"):
        raise ValueError("Expected the public, ungated GLM checkpoint metadata")
    entries = [entry for entry in metadata["siblings"] if entry["rfilename"].endswith(".safetensors")]
    expected = {entry["rfilename"]: entry for entry in entries}
    receipts = {entry["file"]: entry for entry in manifest["files"] if entry["file"].endswith(".safetensors")}
    index = read(path / "model.safetensors.index.json")
    if len(expected) != WEIGHT_SHARDS or set(expected) != set(receipts) or set(expected) != set(index["weight_map"].values()):
        raise ValueError("GLM download receipts, public metadata, and index must identify all 48 weight shards")
    actual_bytes = 0
    verified = []
    for name, entry in sorted(expected.items()):
        if Path(name).name != name:
            raise ValueError("Unexpected nested weight path")
        file = path / name
        actual = file.stat().st_size
        if actual != entry["size"] or receipts[name].get("bytes") != actual:
            raise ValueError("Actual GLM shard size differs from download provenance: " + name)
        published = entry.get("lfs", {}).get("sha256")
        if not published or len(published) != 64:
            raise ValueError("GLM shard lacks a published SHA256: " + name)
        receipt = receipts[name].get("sha256")
        rehashed = verify_all_hashes or not receipt
        digest = sha(file) if rehashed else receipt
        if digest != published:
            raise ValueError("GLM shard hash receipt or actual hash differs from published SHA256: " + name)
        verified.append(dict(file=name, bytes=actual, sha256=digest,
                             verification="actual_file_rehashed" if rehashed else "verified_download_receipt"))
        actual_bytes += actual
    if actual_bytes != WEIGHT_BYTES:
        raise ValueError(f"Actual GLM weight files total {actual_bytes} bytes; expected {WEIGHT_BYTES}")
    cfg = read(path / "config.json")
    if cfg.get("architectures") != ["Glm4MoeLiteForCausalLM"] or cfg.get("dtype") != "bfloat16":
        raise ValueError("Unexpected GLM architecture or weight dtype")
    template = (path / "chat_template.jinja").read_text()
    if "enable_thinking" not in template:
        raise ValueError("Pinned GLM chat template must support disabling thinking")
    return dict(download_manifest_sha256=sha(manifest_path), download_metadata_sha256=sha(metadata_path),
                config_sha256=sha(path / "config.json"), chat_template_sha256=sha(path / "chat_template.jinja"),
                actual_weight_bytes=actual_bytes, actual_weight_shards=len(verified),
                index_total_size_ignored=index.get("metadata", {}).get("total_size"),
                hash_verification=verified)


def telemetry():
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    gpus = []
    for row in csv.reader(query.stdout.splitlines()):
        if not row:
            continue
        values = [value.strip() for value in row]
        if len(values) != 6:
            raise ValueError("Could not interpret nvidia-smi GPU telemetry")
        gpus.append(dict(index=int(values[0]), uuid=values[1], name=values[2],
                         total_memory_MiB=int(values[3]), used_memory_MiB=int(values[4]),
                         utilization_percent=int(values[5])))
    apps = []
    for row in csv.reader(processes.stdout.splitlines()):
        if not row:
            continue
        values = [value.strip() for value in row]
        if len(values) != 3 or not values[0].isdigit():
            raise ValueError("Could not certify the current GPU process list")
        apps.append(dict(pid=int(values[0]), gpu_uuid=values[1], used_memory_MiB=values[2]))
    return dict(utc=utc().isoformat(), gpus=gpus, compute_processes=apps)


def require_idle(snapshot, gpu_ids):
    lookup = {gpu["index"]: gpu for gpu in snapshot["gpus"]}
    for index in gpu_ids:
        if index not in lookup:
            raise ValueError("GPU is absent from telemetry: " + str(index))
        gpu = lookup[index]
        if gpu["total_memory_MiB"] < 79000:
            raise ValueError("GLM BF16 replicas require an 80 GB class GPU")
        active = any(app["gpu_uuid"] == gpu["uuid"] for app in snapshot["compute_processes"])
        if active or gpu["used_memory_MiB"] > 512 or gpu["utilization_percent"] > 5:
            raise RuntimeError(f"GPU {index} is not idle; refusing to displace another workload")


def require_free_ports(ports):
    for port in ports:
        with socket.socket() as connection:
            connection.settimeout(.5)
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is occupied; refusing to replace a local service")


def server_command(model_path, port, memory_utilization=.90, compiled=False):
    command = [str(ROOT / ".venv/bin/vllm"), "serve", str(model_path), "--served-model-name", WRITER,
               "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16", "--tensor-parallel-size", "1",
               "--max-model-len", "16384", "--max-num-seqs", "2", "--gpu-memory-utilization", str(memory_utilization),
               "--attention-backend", "TRITON_MLA", "--kv-cache-dtype", "bfloat16", "--reasoning-parser", "glm45",
               "--default-chat-template-kwargs", '{"enable_thinking":false}']
    if not compiled:
        command.append("--enforce-eager")
    return command


def ready(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as stream:
            return WRITER in {model["id"] for model in json.load(stream).get("data", [])}
    except (OSError, ValueError):
        return False


def terminate(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/workspace/models/GLM-4.7-Flash"))
    parser.add_argument("--ports", nargs=2, type=int, default=[18000, 18001])
    parser.add_argument("--gpus", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--gpu-memory-utilization", type=float, default=.90)
    parser.add_argument("--ready-timeout-minutes", type=float, default=20)
    parser.add_argument("--verify-weights-sha256", action="store_true", help="Rehash every weight file even when its download receipt has a verified hash")
    parser.add_argument("--compiled", action="store_true", help="Opt into compilation; eager startup is the default")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.ports)) != 2 or any(not 1 <= p <= 65535 for p in args.ports):
        parser.error("Require two distinct TCP ports")
    if len(set(args.gpus)) != 2 or any(gpu < 0 for gpu in args.gpus):
        parser.error("Require two distinct GPU indices")
    if not .8 <= args.gpu_memory_utilization <= .95 or not 1 <= args.ready_timeout_minutes <= 30:
        parser.error("Memory utilization must be .8..95 and readiness timeout 1..30 minutes")
    commands = [server_command(args.model_path, port, args.gpu_memory_utilization, args.compiled) for port in args.ports]
    if not args.execute:
        print(json.dumps(dict(event="dry_run", model=WRITER, revision=REVISION, precision="bfloat16",
            eager=not args.compiled, ports=args.ports, gpus=args.gpus, commands=commands,
            weight_bytes_required=WEIGHT_BYTES, cutoff_utc=CUTOFF.isoformat(),
            no_download=True, no_inference=True), indent=2))
        return
    if utc() >= CUTOFF:
        raise TimeoutError("The optional GLM writer cutoff has passed; refusing to start servers")
    output = ROOT / "out/overnight20261021"
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "glm47_start.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    serving_version = importlib.metadata.version("vllm")
    if serving_version.split("+")[0] != "0.31.0":
        raise ValueError("This optional launcher was checked against vLLM 0.31.0; inspect a changed serving version first")
    metadata_path = output / "glm47_servers.json"
    if metadata_path.exists():
        previous = read(metadata_path)
        for record in previous.get("servers", []):
            try:
                os.kill(record["pid"], 0)
            except ProcessLookupError:
                continue
            raise RuntimeError("A previous managed GLM server PID may still be alive; inspect it before restarting")
    checkpoint = verify_checkpoint(args.model_path, args.verify_weights_sha256)
    initial = telemetry()
    require_idle(initial, args.gpus)
    require_free_ports(args.ports)
    result = dict(started_utc=utc().isoformat(), model=WRITER, revision=REVISION, precision="bfloat16",
                  architecture="Glm4MoeLiteForCausalLM", reasoning_parser="glm45", attention_backend="TRITON_MLA",
                  enable_thinking=False, eager=not args.compiled, servers=[], checkpoint=checkpoint,
                  telemetry_before=initial, status="starting", cutoff_utc=CUTOFF.isoformat(),
                  vllm_version=serving_version, transformers_version=importlib.metadata.version("transformers"))
    children = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for gpu, port, command in zip(args.gpus, args.ports, commands):
            if utc() >= CUTOFF:
                raise TimeoutError("Optional writer cutoff reached before replica launch")
            require_idle(telemetry(), [gpu])
            require_free_ports([port])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                       VLLM_NO_USAGE_STATS="1", TORCHINDUCTOR_CACHE_DIR=f"/workspace/.cache/inductor_glm47_{gpu}",
                       PATH=str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", ""))
            log = output / f"glm47_server_gpu{gpu}.log"
            with log.open("a") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            result["servers"].append(dict(gpu=gpu, port=port, pid=child.pid, command=command, log=str(log)))
            atomic(metadata_path, result)
        until = min(CUTOFF.timestamp(), time.time() + args.ready_timeout_minutes * 60)
        while time.time() < until:
            snapshot = telemetry()
            with (output / "glm47_startup_telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(snapshot) + "\n")
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A GLM replica exited before readiness; inspect its server log")
            if all(ready(port) for port in args.ports):
                result.update(status="ready", ready_utc=utc().isoformat(), telemetry_ready=snapshot)
                atomic(metadata_path, result)
                print(json.dumps(dict(event="glm_servers_ready", metadata=str(metadata_path),
                    pids=[c.pid for c in children], model=WRITER, no_inference_performed=True)), flush=True)
                return
            time.sleep(min(10, max(.1, until - time.time())))
        raise TimeoutError("GLM replicas did not become ready before the startup deadline")
    except BaseException as exc:
        for child in children:
            terminate(child)
        result.update(status="startup_failed", error=repr(exc), stopped_utc=utc().isoformat())
        atomic(metadata_path, result)
        raise


if __name__ == "__main__":
    main()
