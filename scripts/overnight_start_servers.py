"""Start one pinned local writer replica per GPU for the authorized study."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/Qwen3.5-9B")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    ap.add_argument("--ports", nargs=2, type=int, default=[8000, 8001])
    ap.add_argument("--tag", default="qwen9b")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "out/overnight20261021"
    output.mkdir(parents=True, exist_ok=True)
    model = json.loads((Path(args.model_path) / "download_manifest.json").read_text())
    if model["model"] != args.model_id or model["revision"] != args.revision:
        raise ValueError("Checkpoint provenance differs from the requested model")
    records = []
    for gpu, port in enumerate(args.ports):
        command = [str(root / ".venv/bin/vllm"), "serve", args.model_path,
                   "--served-model-name", args.model_id, "--host", "127.0.0.1",
                   "--port", str(port), "--dtype", "bfloat16", "--max-model-len", "16384",
                   "--max-num-seqs", "2", "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                   "--language-model-only", "--reasoning-parser", "qwen3"]
        if args.enforce_eager:
            command.append("--enforce-eager")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                   TOKENIZERS_PARALLELISM="false", VLLM_NO_USAGE_STATS="1",
                   PATH=str(root / ".venv/bin") + os.pathsep + os.environ.get("PATH", ""),
                   TORCHINDUCTOR_CACHE_DIR=f"/workspace/.cache/inductor_{args.tag}_{gpu}")
        log = output / f"{args.tag}_server_gpu{gpu}.log"
        with log.open("w") as stream:
            child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        records.append(dict(gpu=gpu, port=port, pid=child.pid, command=command, log=str(log)))
    result = dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), servers=records,
                  model=args.model_id, revision=args.revision, precision="bfloat16")
    (output / f"{args.tag}_servers.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
