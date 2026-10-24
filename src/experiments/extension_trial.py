"""Budgeted local trials, launched through experiments/main.py.

Uses the original UQ task, prompt builder and evolutionary population. Search
never loads test data; every valid program gets the same validation evaluation.
"""
from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import logging
import multiprocessing as mp
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from omegaconf import OmegaConf


def _write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _eval_worker(config, code, imports, split, connection):
    """Separate process permits a hard per-candidate timeout."""
    try:
        import torch
        from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import (
            evaluate_func, generate_input,
        )
        from packing.logging.function_class import FunctionClass

        torch.set_num_threads(1)
        logging.getLogger().setLevel(logging.ERROR)
        cfg = OmegaConf.create(config)
        def one_split(name):
            keep_scores = name == "val" or (name == "train" and cfg.trial.get("save_train_predictions", False))
            OmegaConf.update(cfg, "task.keep_predictions", bool(keep_scores), force_add=True)
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            started = time.monotonic()
            data = generate_input(cfg.task, name)
            fc = evaluate_func(cfg, data, FunctionClass(code, imports))
            if fc.fail_flag or not np.isfinite(fc.score):
                raise ValueError(fc.extra_metrics.get("error", "invalid candidate score"))
            result = {"score": float(fc.score), "metrics": fc.extra_metrics,
                      "seconds": time.monotonic() - started}
            if keep_scores:
                result["scores"] = fc.eval.output_struct.tolist()
            return result
        if isinstance(split, list):
            result = {}
            for name in split:
                try:
                    result[name] = one_split(name)
                except Exception as exc:
                    result[name] = {"error": f"{type(exc).__name__}: {exc}"}
                    break
            connection.send(result)
        else:
            connection.send(one_split(split))
    except BaseException as exc:
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def evaluate_candidate(cfg, code, imports, split, timeout=None):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    proc = context.Process(target=_eval_worker, args=(
        OmegaConf.to_container(cfg, resolve=True), code, imports, split, child,
    ))
    started = time.monotonic()
    proc.start()
    child.close()
    limit = float(timeout if timeout is not None else cfg.task.timeout_period)
    try:
        import psutil
        watched = psutil.Process(proc.pid)
        memory_limit = float(cfg.task.mem_limit_gb) * 1024**3
        while not parent.poll(0.2):
            if time.monotonic() - started >= limit:
                return {"error": f"evaluation timeout after {limit:.1f}s"}
            try:
                if watched.memory_info().rss > memory_limit:
                    return {"error": f"candidate exceeded {cfg.task.mem_limit_gb} GiB RSS"}
            except psutil.NoSuchProcess:
                break
        try:
            result = parent.recv()
        except EOFError:
            result = {"error": f"evaluation worker exited ({proc.exitcode})"}
        result["seconds"] = time.monotonic() - started
        return result
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        parent.close()


def build_prompt(cfg, database, baseline, baseline_score, round_index):
    from packing.model.prompt import generate_prompt, programs_to_prompt_creative
    from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import system_prompt_fabric

    mode = cfg.search_mode
    random.seed(int(cfg.seed) + round_index)
    np.random.seed(int(cfg.seed) + round_index)
    if mode == "frozen_context":
        manifest = json.loads(Path(cfg.trial.frozen_contexts_path).read_text())
        contexts = manifest["contexts"]
        if int(cfg.num_rounds) != len(contexts):
            raise ValueError("Frozen-context rounds must equal the manifest context count")
        return contexts[round_index]["chat"], None
    elif mode in {"evolution", "structured"}:
        fraction = round_index / max(1, int(cfg.num_rounds))
        percentile = cfg.initial_percentile + fraction * (cfg.final_percentile - cfg.initial_percentile)
        chat, _, island, _, _ = generate_prompt(cfg, database, percentile)
    elif mode == "baseline_seeded":
        chat, _ = programs_to_prompt_creative(cfg, [baseline], [baseline_score], [{}])
        island = None
    elif mode == "independent":
        chat = [{"role": "system", "content": system_prompt_fabric()},
                {"role": "user", "content": (
                    "Design a detector from scratch. Required evaluator interface:\n"
                    "- class HallucinationDetector(Estimator); Estimator is provided, so do not import or redefine it.\n"
                    "- def __init__(self, context): call super().__init__(context, "
                    "stats_dependencies=[\"sequence_logprobs\", \"top_k_logits\", \"top_k_indices\"], level=\"sequence\").\n"
                    "- def __call__(self, stats): return a finite numpy array of shape [N], one uncertainty score per sample.\n"
                    "stats is a dictionary of batch lists, using the array shapes in the system prompt. "
                    "Implement __call__ directly; the evaluator does not invoke methods named score, estimate, or compute."
                )}]
        island = None
    else:
        raise ValueError(f"Unknown search_mode: {mode}")
    if mode == "structured":
        families = [
            "Use positional weighting of token surprisal; avoid top-k shape features.",
            "Use confidence gaps between the observed token and the top alternatives.",
            "Use contiguous runs or local windows of high token surprisal.",
            "Use distributional shape or concentration of top-k probabilities.",
        ]
        chat[0]["content"] += "\nRequired method family: " + families[round_index % len(families)]
    return chat, int(island) if island is not None else None


def parse_candidate(content):
    from packing.utils.functions import extract_class
    code = extract_class(content, "HallucinationDetector", base_class="Estimator")
    if not code:
        raise ValueError("No HallucinationDetector class in response")
    # Extract imports from the selected fenced code only, using AST so imports
    # inside methods remain inside methods. Narrative text is never executed.
    import re
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
    source = next((b for b in blocks if code.strip() in b), content if not blocks else code)
    try:
        ast.parse(source)
    except SyntaxError:
        source = code
    tree = ast.parse(source)
    imports = "\n".join(ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    ast.parse(code)
    return code, imports


def run_trial(cfg):
    from openai import OpenAI
    from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import get_initial_func
    from packing.funsearch.programs_database import ProgramsDatabase
    from packing.utils.functions import function_to_string
    from experiments.prompt_policies import candidate_prompt, POLICIES

    OmegaConf.set_struct(cfg, False)
    if cfg.task.supervised:
        raise ValueError("Trial protocol currently supports unsupervised candidates only")
    endpoint = urlparse(cfg.cluster.api_base_url)
    if cfg.trial.local_only and endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Local-only trial requires a loopback model endpoint")
    if cfg.search_mode not in {"evolution", "independent", "baseline_seeded", "structured", "frozen_context"}:
        raise ValueError("Invalid search mode")
    policy = cfg.trial.get("prompt_policy", "original")
    if policy not in POLICIES:
        raise ValueError("Invalid prompt policy")
    if cfg.search_mode == "structured" and policy != "original":
        raise ValueError("Test the structured mode and prompt-policy intervention separately")
    cfg.trial.prompt_policy = policy
    if cfg.search_mode == "frozen_context":
        context_bytes = Path(cfg.trial.frozen_contexts_path).read_bytes()
        digest = hashlib.sha256(context_bytes).hexdigest()
        if cfg.trial.get("frozen_contexts_sha256", digest) != digest:
            raise ValueError("Frozen-context manifest hash differs from the planned artifact")
        cfg.trial.frozen_contexts_sha256 = digest
    if min(int(cfg.num_rounds), int(cfg.num_outputs_per_prompt), int(cfg.trial.max_total_tokens)) <= 0:
        raise ValueError("Round, output and token budgets must be positive")
    train_path = Path(cfg.task.train_raw_path)
    if not train_path.is_file():
        raise FileNotFoundError(f"Precompute training features first: {train_path}")
    cfg.function_str_to_extract = "HallucinationDetector"
    cfg.code_type = "class"
    cfg.task.bootstrap_samples = 0
    cfg.trial.interface_contract_version = 2 if cfg.search_mode == "independent" else 1
    output = Path(cfg.logs_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "candidates.jsonl"
    if records_path.exists() and not cfg.trial.resume:
        raise FileExistsError(f"Run exists; use a new run name or trial.resume=true: {output}")
    saved_config = OmegaConf.to_container(cfg, resolve=True)
    # Local endpoint key is a placeholder; never persist an actual credential.
    saved_config["cluster"]["api_key"] = "<redacted>"
    if "api_key" in saved_config:
        saved_config["api_key"] = "<redacted>"
    config_path = output / "trial_config.json"
    if cfg.trial.resume and config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("trial", {}).get("interface_contract_version", 1) != saved_config["trial"]["interface_contract_version"]:
            raise ValueError("Cannot resume with a changed interface contract")
        for key in ("model", "task", "search_mode", "seed"):
            if previous[key] != saved_config[key]:
                raise ValueError(f"Cannot resume with a changed {key}")
        for key, default in [("prompt_policy", "original"), ("frozen_contexts_sha256", None),
                             ("save_train_predictions", False)]:
            if previous.get("trial", {}).get(key, default) != saved_config["trial"].get(key, default):
                raise ValueError(f"Cannot resume with a changed trial.{key}")
    _write_json(config_path, saved_config)
    deadline = time.monotonic() + float(cfg.trial.max_seconds)

    def evaluate(code, imports, split):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"error": "run time budget exhausted"}
        return evaluate_candidate(cfg, code, imports, split, min(remaining, cfg.task.timeout_period))

    initial, _, _ = get_initial_func(cfg.task)
    baseline = function_to_string(initial)
    baseline_results = evaluate(baseline, "", ["train", "val"])
    baseline_train = baseline_results.get("train", {"error": baseline_results.get("error", "missing train result")})
    baseline_val = baseline_results.get("val", {"error": baseline_results.get("error", "missing val result")})
    if "error" in baseline_train or "error" in baseline_val:
        raise RuntimeError(f"Baseline evaluation failed: {baseline_train}, {baseline_val}")
    if "scores" in baseline_train:
        np.save(output / "baseline_train_scores.npy", np.asarray(baseline_train.pop("scores"), dtype=np.float32))
    np.save(output / "baseline_val_scores.npy", np.asarray(baseline_val.pop("scores"), dtype=np.float32))
    database = ProgramsDatabase(cfg.programdatabaseConfig)
    database.register_program(baseline, baseline_train["score"], None, {})
    records = [json.loads(x) for x in records_path.read_text().splitlines()] if records_path.exists() else []
    seen = set()
    for row in records:
        if row.get("code_hash"):
            seen.add(row["code_hash"])
        if "score" in row.get("train", {}):
            database.register_program(row["imports"] + "\n" + row["code"], row["train"]["score"], row["island"], {})
    spent = sum(int(r.get("usage", {}).get("total_tokens") or 0) for r in records)
    starting_round = max((r["round"] for r in records), default=-1) + 1
    failed_rounds = 0
    stop_reason = "round_budget"
    client = OpenAI(base_url=cfg.cluster.api_base_url, api_key="EMPTY", max_retries=0,
                    timeout=float(cfg.trial.request_timeout_seconds))

    def generate(chat, request_seed):
        started = time.monotonic()
        try:
            extra = {"top_k": int(cfg.model.topk)}
            if cfg.model.get("supports_chat_template_kwargs", True):
                extra["chat_template_kwargs"] = {"enable_thinking": bool(cfg.trial.enable_thinking)}
            elif cfg.trial.enable_thinking:
                raise ValueError("This native tokenizer does not support the requested thinking toggle")
            response = client.chat.completions.create(
                model=cfg.model.hf_model_id, messages=chat,
                temperature=float(cfg.model.temperature), top_p=float(cfg.model.topp),
                max_tokens=int(cfg.model.max_tokens), seed=request_seed,
                extra_body=extra,
                timeout=max(1.0, min(float(cfg.trial.request_timeout_seconds), deadline - time.monotonic())),
            )
            usage = response.usage.model_dump() if response.usage else {}
            return {"content": response.choices[0].message.content or "", "usage": usage,
                    "finish_reason": response.choices[0].finish_reason,
                    "generation_seconds": time.monotonic() - started}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "usage": {},
                    "generation_seconds": time.monotonic() - started}

    for round_index in range(starting_round, int(cfg.num_rounds)):
        if time.monotonic() >= deadline:
            stop_reason = "time_budget"
            break
        # The local server has a 16K total context cap. Reserve the worst case
        # for the entire request batch before issuing any generation.
        if spent + int(cfg.num_outputs_per_prompt) * 16384 > int(cfg.trial.max_total_tokens):
            stop_reason = "token_budget"
            break
        chat, island = build_prompt(cfg, database, baseline, baseline_train["score"], round_index)
        _write_json(output / f"prompt_{round_index:04d}.json", chat)
        request_chats = [candidate_prompt(chat, policy, round_index, index)
                         for index in range(int(cfg.num_outputs_per_prompt))]
        _write_json(output / f"candidate_prompts_{round_index:04d}.json", request_chats)
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg.num_outputs_per_prompt)) as pool:
            responses = list(pool.map(lambda i: generate(request_chats[i], int(cfg.seed) + round_index * 100 + i),
                                      range(int(cfg.num_outputs_per_prompt))))
        _write_json(output / f"responses_{round_index:04d}.json", responses)
        spent += sum(int(response["usage"].get("total_tokens") or 0) for response in responses)
        round_successes = 0
        rows, futures = [], {}
        # Candidate programs run in independent spawned processes. Evaluate the
        # already generated batch concurrently, then register results in their
        # original order so population updates remain deterministic.
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg.trial.get("evaluation_workers", 2))) as pool:
            for index, response in enumerate(responses):
                row = {"round": round_index, "index": index, "island": island, **response}
                row.update(prompt_policy=policy, request_seed=int(cfg.seed) + round_index * 100 + index,
                           prompt_sha256=hashlib.sha256(json.dumps(request_chats[index], sort_keys=True).encode()).hexdigest())
                rows.append(row)
                if "error" in row:
                    continue
                try:
                    code, imports = parse_candidate(row["content"])
                    digest = hashlib.sha256((imports + "\n" + code).encode()).hexdigest()
                    row.update(code=code, imports=imports, code_hash=digest, duplicate=digest in seen)
                    if digest in seen:
                        row["error"] = "exact duplicate code"
                    else:
                        seen.add(digest)
                        futures[index] = pool.submit(evaluate, code, imports, ["train", "val"])
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
            for index, future in futures.items():
                row = rows[index]
                try:
                    results = future.result()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    continue
                row["evaluation_seconds"] = results.get("seconds", 0.0)
                row["train"] = results.get("train", {"error": results.get("error", "missing training result")})
                if "scores" in row["train"]:
                    train_score_path = output / f"train_scores_{row['code_hash']}.npy"
                    np.save(train_score_path, np.asarray(row["train"].pop("scores"), dtype=np.float32))
                    row["train_scores_file"] = str(train_score_path)
                if "score" in row["train"]:
                    row["val"] = results.get("val", {"error": "validation evaluation did not finish"})
                    if "scores" in row["val"]:
                        score_path = output / f"val_scores_{row['code_hash']}.npy"
                        np.save(score_path, np.asarray(row["val"].pop("scores"), dtype=np.float32))
                        row["val_scores_file"] = str(score_path)
                    database.register_program(row["imports"] + "\n" + row["code"], row["train"]["score"], island, {})
        for row in rows:
            if "score" in row.get("val", {}):
                round_successes += 1
            row["cumulative_tokens"] = spent
            row["cumulative_seconds"] = float(cfg.trial.max_seconds) - max(0.0, deadline - time.monotonic())
            with records_path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            records.append(row)
        eligible = [r for r in records if "score" in r.get("val", {})]
        winner = max(eligible, key=lambda r: r["val"]["score"], default=None)
        summary = {"status": "running", "rounds_completed": round_index + 1,
                   "requests": len(records), "valid_candidates": len(eligible), "total_tokens": spent,
                   "elapsed_seconds": float(cfg.trial.max_seconds) - max(0.0, deadline - time.monotonic()),
                   "baseline_train": baseline_train, "baseline_val": baseline_val,
                   "best_validation_auc_percent": max(baseline_val["score"], winner["val"]["score"] if winner else 0),
                   "held_out_test_evaluated": False}
        _write_json(output / "summary.json", summary)
        logging.info("Trial progress: %s", summary)
        failed_rounds = 0 if round_successes else failed_rounds + 1
        if failed_rounds >= int(cfg.trial.max_consecutive_failed_rounds):
            stop_reason = "consecutive_failed_rounds"
            break
    eligible = [r for r in records if "score" in r.get("val", {})]
    winner = max(eligible, key=lambda r: r["val"]["score"], default=None)
    if winner is None or baseline_val["score"] >= winner["val"]["score"]:
        code, imports, winner_id, val = baseline, "", "baseline", baseline_val["score"]
    else:
        code, imports = winner["code"], winner["imports"]
        winner_id, val = winner["code_hash"], winner["val"]["score"]
    (output / "frozen_estimator.py").write_text(imports + "\n" + code + "\n")
    summary = {"status": "finished", "stop_reason": stop_reason, "requests": len(records),
               "valid_candidates": len(eligible), "total_tokens": spent,
               "baseline_train": baseline_train, "baseline_val": baseline_val,
               "selected_candidate": winner_id, "selected_validation_auc_percent": val,
               "held_out_test_evaluated": False,
               "elapsed_seconds": float(cfg.trial.max_seconds) - max(0.0, deadline - time.monotonic())}
    shortlist_size = int(cfg.trial.get("shortlist_size", 1))
    if shortlist_size > 1:
        shortlist = [(baseline_val["score"], "baseline", baseline, "")]
        shortlist += [(r["val"]["score"], r["code_hash"], r["code"], r["imports"]) for r in eligible]
        shortlist.sort(key=lambda item:-item[0])
        directory = output / "validation_shortlist"
        directory.mkdir(exist_ok=True)
        exported = []
        for rank, (score, digest, source, source_imports) in enumerate(shortlist[:shortlist_size], 1):
            path = directory / f"rank_{rank:02d}_{digest[:12]}.py"
            path.write_text(source_imports + "\n" + source + "\n")
            exported.append(dict(rank=rank, candidate=digest, validation_auc_percent=score,
                                 file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        summary["validation_shortlist"] = exported
    _write_json(output / "summary.json", summary)
    logging.info("Trial finished: %s", summary)
