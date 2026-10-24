"""Additive CPU-only transfer of immutable C8741 source-fitted normalization.

Default prepares a plan only. Never invokes the source helper's fit/CLI, changes
its artifacts, extracts features, or touches a GPU. Six results remain immutable
when the three formally guarded target additions become available.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from scripts import overnight_frozen_normalization as frozen
from scripts import overnight_transfer_eval as provenance

ROOT = Path(__file__).resolve().parents[1]
NIGHT = ROOT / "results/overnight20261021"
OUTPUT = NIGHT / "normalization_transfer"
SOURCE = NIGHT / "frozen_train_normalization"
SOURCE_SHA = "a96fdf5af49a1a6f0f6426d79e2ef60b596be1d7ffd38a672afb6338523cc884"
HELPER_SHA = "f2f92a85857801c301e1383fabde5b4903c68a401f46d7a6dbb1a968a7fdc89d"
WEIGHTS_SHA = "e83b7b1e83b6a2e347f0d080bf50bb5d50d67b6ae7f578ce40251f5e94340341"
DEADLINE = dt.datetime(2026, 10, 22, 2, 3, tzinfo=dt.timezone.utc)
MAX_SECONDS = 1200
COUNT = 500
CASES = ("original_1", "original_128", "original_512", "shuffle451_512")
ORIGINAL, FIXED, NLL = "original_C8741_512", "source_fixed_C8741", "canonical_nll"
METHODS = (ORIGINAL, FIXED, NLL)
ROW_KEYS = ("labels", "claim_group_ids", "duplicate_disjoint_mask")
sha = provenance.sha


def now():
    return dt.datetime.now(dt.timezone.utc)


def read(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Symlinked receipt refused: " + str(path))
    return json.loads(path.read_text()) if path.exists() else None


def dump(path, data):
    frozen.atomic_json(path, data)


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def local(path):
    p = Path(path)
    if p.is_absolute():
        for root in (Path("/workspace/mod1"), ROOT):
            try:
                return ROOT / p.relative_to(root)
            except ValueError:
                pass
        raise ValueError("Artifact outside repository")
    if ".." in p.parts:
        raise ValueError("Artifact path escapes repository")
    return ROOT / p


def assert_protected(plan):
    for name, digest in plan["protected_sha256"].items():
        p = local(name)
        if p.is_symlink() or not p.is_file() or sha(p) != digest:
            raise ValueError("Protected source artifact changed: " + name)


def lp_dependency_contract():
    tree = ast.parse(frozen.parent.SOURCE.read_text())
    accesses = [n for n in ast.walk(tree) if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == "stats"]
    if not accesses or any(not isinstance(n.slice, ast.Constant) or n.slice.value != "sequence_logprobs" for n in accesses):
        raise ValueError("C8741 no longer has an LP-only dependency")
    return frozen.source_contract()


def prepare():
    if sha(frozen.parent.SOURCE) != SOURCE_SHA or sha(Path(frozen.__file__)) != HELPER_SHA or sha(SOURCE / "normalization.json") != WEIGHTS_SHA:
        raise ValueError("Pinned source, inference helper or fitted statistics changed")
    weights, fit, report = [read(SOURCE / x) for x in ("normalization.json", "FIT_RECEIPT.json", "FROZEN_NORMALIZATION_REPORT.json")]
    if not (weights["train_only"] is True and weights["source_fit_rows"] == 5605 and weights["excluded_validation_rows"] == 1402
            and weights["coefficients"] == [.4, .25, .2, .15] and weights["mad_multiplier"] == 1.4826
            and fit["weights_sha256"] == WEIGHTS_SHA and fit["fitting_labels_used"] is False
            and report["status"] == "complete_exploratory_train_only_control" and report["all9_original_proxy_primary_exact"]
            and report["all36_frozen_scores_chunk_order_byte_invariant"]):
        raise ValueError("Completed source-fit contract missing")
    base = read(NIGHT / "batch_sensitivity/PLAN.json")
    archive = read(ROOT / "data/cache_backup_manifest.json")
    six = sorted(Path(x).name.removesuffix(".pt.gz") for x in archive if Path(x).parent.name == "mod1-transfer-cache")
    if len(six) != 6:
        raise ValueError("Expected exactly six historical target caches")
    datasets = []
    protected = {rel(local(p)): h for p, h in read(SOURCE / "PLAN.json")["protected_source_sha256"].items()}
    additions = [Path(__file__), Path(provenance.__file__), SOURCE / "PLAN.json", SOURCE / "normalization.json",
                 SOURCE / "FIT_RECEIPT.json", SOURCE / "FROZEN_NORMALIZATION_REPORT.json", SOURCE / "fit_features_and_split.npz",
                 NIGHT / "batch_sensitivity/PLAN.json", ROOT / "data/cache_backup_manifest.json",
                 NIGHT / "restored_caches.json", ROOT / "out/campaign/transfer_llama32_1b/protocol.json",
                 ROOT / "src/experiments/extension_trial.py",
                 ROOT / "src/packing/evaluate/uncertainty_quantification/found_formulas.py",
                 ROOT / "src/packing/evaluate/uncertainty_quantification/estimator_base.py"]
    for row in base["datasets"]:
        name = row["name"]
        point = SOURCE / "frozen" / (name + "_scores.npz")
        result = SOURCE / "frozen" / (name + ".json")
        entry = dict(index=row["index"], name=name, rows=row["rows"], csv=rel(local(row["csv"])),
                     csv_sha256=row["csv_sha256"], source_cache=rel(local(row["cache"])), source_raw_sha256=row["raw_sha256"],
                     source_scores=rel(point), historical_six=name in six)
        additions.extend([point, result, local(row["csv"])])
        if name in six:
            reference = NIGHT / "transfer_llama32_1b/aa4771d92b2c7bca/new" / (name + "_scores.npz")
            entry["six_nll_reference"] = rel(reference)
            additions.extend([reference, reference.with_name(name + "_bootstrap.npz")])
        datasets.append(entry)
    if len(datasets) != 9 or {x["index"] for x in datasets if not x["historical_six"]} != {6, 7, 8}:
        raise ValueError("Unexpected nine-dataset ordering/additions")
    additions.append(local(base["training_csv"]))
    for p in additions:
        protected[rel(p)] = sha(p)
    plan = dict(format="source_fixed_C8741_transfer_v1", created_utc=now().isoformat(),
                source_sha256=SOURCE_SHA, frozen_helper_sha256=HELPER_SHA, weights_sha256=WEIGHTS_SHA,
                protected_sha256=protected, datasets=datasets, six_names=six, methods=list(METHODS),
                source_contract=lp_dependency_contract(), source_scoring="LP-only must reproduce saved all-nine original/fixed/NLL bytes before target scoring",
                target_cases=list(CASES), original_scoring_chunk_size=512, source_fitting=False, target_fitting=False,
                source_statistics_refit=False, source_validation_selection_already_frozen=True, coefficients_changed=False,
                source_artifacts_and_six_results_immutable=True, bootstrap_samples=COUNT, bootstrap_unit="claim",
                bootstrap_seed="441 + int(sha256(dataset_name)[:8],16); strict adds1000003; shared draws across methods/models",
                workers=4, cumulative_active_seconds_limit=MAX_SECONDS, deadline_utc=DEADLINE.isoformat(), gpu_calls=0,
                completion_receipt="results/overnight20261021/remaining_transfer_completion.json",
                scientific_limits=["Exploratory after source benchmark viewing; no primary replacement or target-based selection",
                                   "Source zero-MAD rule keeps f3/f4 disabled on target; medians do not establish calibration transfer",
                                   "Shared original tokenizer; not native-tokenizer transfer"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old = read(OUTPUT / "PLAN.json")
    if old and any(old.get(k) != v for k, v in plan.items() if k != "created_utc"):
        raise ValueError("Prepared plan changed; refusing overwrite")
    if old is None:
        dump(OUTPUT / "PLAN.json", plan)
    return old or plan


def load_three(meta):
    from scripts.evaluate_cached_claims import load_estimators
    # Separate exec namespaces are essential: ProxiedEstimator swaps its own
    # candidate namespace's np only; the original can never share that mapping.
    a, _ = load_estimators([frozen.parent.SOURCE], meta)
    b, _ = load_estimators([frozen.parent.SOURCE], meta)
    original = a[frozen.parent.METHOD]
    proxy_candidate = b[frozen.parent.METHOD]
    if original.__call__.__globals__ is proxy_candidate.__call__.__globals__:
        raise ValueError("Original/proxy candidates share a compiled namespace")
    weights = read(SOURCE / "normalization.json")
    if sha(SOURCE / "normalization.json") != WEIGHTS_SHA:
        raise ValueError("Frozen statistics changed")
    return original, frozen.ProxiedEstimator(proxy_candidate, weights["statistics"]), a[frozen.parent.BASELINE]


def rows_from_csv(csv, training):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(csv, usecols=["claim", "is_hallu"])
    names = frame.claim.astype(str).str.strip().tolist()
    normalized = [" ".join(x.casefold().split()) for x in names]
    groups, _ = pd.factorize(np.array(normalized, dtype=object), sort=False)
    train = set(pd.read_csv(training, usecols=["claim"]).claim.fillna("").astype(str).str.split().str.join(" ").str.casefold())
    return names, dict(labels=frame.is_hallu.to_numpy(dtype=int), claim_group_ids=groups,
                       duplicate_disjoint_mask=np.array([x not in train for x in normalized], dtype=bool))


def require_rows(arrays, expected):
    if any(k not in arrays or not frozen.parent.exact(arrays[k], expected[k]) for k in ROW_KEYS):
        raise ValueError("Labels, claim groups or strict masks differ")


def bootstrap_arrays(arrays, name, count=COUNT):
    import numpy as np
    from scripts.evaluate_cached_claims import bootstrap_counts, weighted_auc_bootstrap
    output = {}
    for scope in ("full", "strict"):
        mask = np.ones(len(arrays["labels"]), dtype=bool) if scope == "full" else arrays["duplicate_disjoint_mask"]
        labels = arrays["labels"][mask]
        _, groups = np.unique(arrays["claim_group_ids"][mask], return_inverse=True)
        if not len(labels) or len(np.unique(labels)) != 2:
            raise ValueError("Both classes required in full and strict scopes")
        seed = 441 + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) + (1000003 if scope == "strict" else 0)
        rng = np.random.default_rng(seed)
        prepared = {}
        for method in METHODS:
            pred = arrays[method][mask]
            order = np.argsort(pred, kind="stable")
            prepared[method] = pred, order, np.r_[0, np.flatnonzero(np.diff(pred[order])) + 1]
            output[scope + "/" + method] = np.empty(count)
        for i in range(count):
            counts = bootstrap_counts(groups, rng, "claim")
            for method, (pred, order, starts) in prepared.items():
                output[scope + "/" + method][i] = weighted_auc_bootstrap(labels, pred, counts, order, starts)
    return output


def load_npz(path):
    import numpy as np
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def worker(plan, index, phase, folder):
    import numpy as np
    import torch
    assert_protected(plan)
    torch.set_num_threads(1)
    e = plan["datasets"][index]
    source = phase == "source"
    raw_path = local(e["source_cache"]) if source else ROOT / "data/restored/mod1-transfer-cache" / (e["name"] + ".pt")
    if source:
        expected_sha = e["source_raw_sha256"]
    else:
        audit = read(folder.parent / "cache_audit.json")
        expected_sha = next(x["raw_sha256"] for x in audit["per_dataset"] if x["dataset"] == e["name"])
    if sha(raw_path) != expected_sha:
        raise ValueError("Raw cache hash mismatch")
    raw = torch.load(raw_path, mmap=True, weights_only=False, map_location="cpu")
    claims, identities = rows_from_csv(local(e["csv"]), ROOT / "data/claims/popqa/popqa_train_filtered_10shot.csv")
    if len(raw["samples"]) != len(claims) or any(s["claim"] != c or float(s["label"]) != float(y)
            for s, c, y in zip(raw["samples"], claims, identities["labels"])):
        raise ValueError("Raw claims/labels not in CSV order")
    gold = load_npz(local(e["source_scores"]))
    require_rows(gold, identities)
    original, fixed, nll = load_three(raw["meta"])
    original_scores, canonical = frozen.parent.score_case(original, nll, raw, "original_512")
    # The transparent proxy is a separate namespace too, and must reconstruct
    # exactly the unwrapped original before any frozen-control result is saved.
    from scripts.evaluate_cached_claims import load_estimators
    third, _ = load_estimators([frozen.parent.SOURCE], raw["meta"])
    transparent = frozen.ProxiedEstimator(third[frozen.parent.METHOD])
    transparent_scores, transparent_nll = frozen.parent.score_case(transparent, nll, raw, "original_512")
    if not frozen.parent.exact(original_scores, transparent_scores) or not frozen.parent.exact(canonical, transparent_nll):
        raise ValueError("Transparent proxy changes original predictions")
    values = None
    invariant = []
    for case in (("original_512",) if source else CASES):
        scores, repeated_nll = frozen.parent.score_case(fixed, nll, raw, case)
        if not frozen.parent.exact(canonical, repeated_nll):
            raise ValueError("NLL changes with scoring case")
        if values is None:
            values = scores
        elif not frozen.parent.exact(values, scores):
            raise ValueError("Frozen normalization is not byte-invariant")
        invariant.append(dict(case=case,score_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),nll_byte_exact=True))
    if source and not all(frozen.parent.exact(a, gold[k]) for a, k in
            ((original_scores, "original_primary512"), (values, "original_512"), (canonical, "canonical_nll"))):
        raise ValueError("LP-only source scores fail original/fixed/NLL byte parity")
    arrays = {**identities, ORIGINAL: original_scores, FIXED: values, NLL: canonical}
    if e["historical_six"] and not source:
        reference = load_npz(local(e["six_nll_reference"]))
        require_rows(reference, identities)
        if not frozen.parent.exact(canonical, reference[provenance.BASELINE]):
            raise ValueError("Target NLL differs from historical-six reference")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / (e["name"] + "_scores.npz"), **arrays)
    np.savez_compressed(folder / (e["name"] + "_bootstrap.npz"), **bootstrap_arrays(arrays, e["name"]))
    dump(folder / (e["name"] + ".json"), dict(status="complete", dataset=e["name"], phase=phase,
         rows=len(claims), raw_sha256=expected_sha, weights_sha256=WEIGHTS_SHA, source_sha256=SOURCE_SHA,
         plan_sha256=sha(OUTPUT / "PLAN.json"), scores_sha256=sha(folder / (e["name"] + "_scores.npz")),
         bootstrap_sha256=sha(folder / (e["name"] + "_bootstrap.npz")),
         source_lp_only_saved_prediction_parity=source, transparent_proxy_original_byte_exact=True,
         frozen_cases=invariant, original_chunk_size=512, target_fit=False, bootstrap_samples=COUNT))


def target_audit(plan, coverage, cohort):
    assert_protected(plan)
    entries = plan["datasets"] if coverage == "nine" else [e for e in plan["datasets"] if e["historical_six"]]
    if coverage == "nine":
        completion = read(local(plan["completion_receipt"]))
        if not completion or completion.get("status") != "complete_three_missing_caches":
            raise ValueError("Three target caches are not formally complete")
        if not provenance.completion_numerical_reference_passed(completion):
            raise ValueError("Full-nine requires the passed 427-row numerical guard")
        required_new = {e["name"] for e in plan["datasets"] if not e["historical_six"]}
        new_rows = completion.get("new_caches", [])
        if (completion.get("all_six_historical_caches_unchanged") is not True
                or len(completion.get("preserved_historical_caches", [])) != 6 or len(new_rows) != 3
                or {Path(e["path"]).stem for e in new_rows} != required_new):
            raise ValueError("Formal completion must preserve six caches and add exactly the three prescribed files")
    args = SimpleNamespace(completion_receipt=local(plan["completion_receipt"]), directory=NIGHT, coverage=coverage,
        cache_dir=ROOT / "data/restored/mod1-transfer-cache", source_cache_dir=ROOT / "data/restored/mod1-eval-cache")
    if any(not (args.source_cache_dir / (e["name"] + ".pt")).is_file() for e in entries):
        raise ValueError("All source caches required for exact token-ID parity")
    files = [local(e["csv"]) for e in plan["datasets"]]
    audit = provenance.audit_target_caches(args, [e["name"] for e in entries], files, {e["name"]:e["index"] for e in plan["datasets"]})
    if any(e["hash_matches_restoration_or_pinned_completion"] is not True or e["all_source_chat_and_raw_token_ids_identical"] is not True for e in audit["per_dataset"]):
        raise ValueError("Every cache needs exact raw hash and source/target input parity")
    if coverage == "nine" and audit.get("numerical_reference_preflight_verified") is not True:
        raise ValueError("Formal numerical reference missing")
    if coverage == "nine":
        previous = read(OUTPUT / "six/cache_audit.json")
        old = {e["dataset"]:e["raw_sha256"] for e in previous["per_dataset"]}
        if any(e["raw_sha256"] != old[e["dataset"]] for e in audit["per_dataset"] if e["dataset"] in old):
            raise ValueError("A historical-six raw cache changed")
    path = cohort / "cache_audit.json"
    old = read(path)
    if old and old != audit:
        # A later completion receipt can enrich provenance but cannot change
        # data used by an existing cohort. Never rewrite a six audit.
        if {x["dataset"]:x["raw_sha256"] for x in old["per_dataset"]} != {x["dataset"]:x["raw_sha256"] for x in audit["per_dataset"]}:
            raise ValueError("Target cohort changed")
    elif old is None:
        dump(path, audit)


def copy_once(source, dest, fingerprint):
    if source.is_symlink() or sha(source) != fingerprint or dest.is_symlink():
        raise ValueError("Unsafe or changed six-result copy")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha(dest) != fingerprint:
            raise ValueError("Refusing to overwrite a distinct six-result copy")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        raise ValueError("Unreviewed partial copy")
    shutil.copyfile(source, tmp)
    if sha(tmp) != fingerprint:
        raise ValueError("Six-result copy hash mismatch")
    os.link(tmp, dest)
    tmp.unlink()


def freeze_six():
    files = sorted((OUTPUT / "six").rglob("*"))
    manifest = {rel(p): sha(p) for p in files if p.is_file() and p.suffix != ".log"}
    payload = dict(status="six_results_immutable", files=manifest)
    old = read(OUTPUT / "SIX_IMMUTABLE.json")
    if old and old != payload:
        raise ValueError("Six results changed after completion")
    if old is None:
        dump(OUTPUT / "SIX_IMMUTABLE.json", payload)
    return payload


def assert_six_immutable():
    receipt = read(OUTPUT / "SIX_IMMUTABLE.json")
    if not receipt:
        raise ValueError("Completed immutable six study required first")
    for name, digest in receipt["files"].items():
        if sha(local(name)) != digest:
            raise ValueError("Immutable six artifacts changed")
    return receipt


def reuse_six(plan):
    receipt = assert_six_immutable()
    for e in plan["datasets"]:
        if not e["historical_six"]:
            continue
        for suffix in ("_scores.npz", "_bootstrap.npz", ".json"):
            source = OUTPUT / "six/target" / (e["name"] + suffix)
            copy_once(source, OUTPUT / "nine/target" / source.name, receipt["files"][rel(source)])


class Bounds:
    def __init__(self, deadline, stop=None):
        self.deadline, self.stop = deadline, stop or threading.Event()

    def check(self):
        if self.stop.is_set() or any((NIGHT / p).exists() for p in
                ("watchdog.cancel", "manual_stop", "manual_stop.json", "normalization_transfer.cancel")):
            raise InterruptedError("Normalization transfer cancelled")
        if now() >= self.deadline:
            raise TimeoutError("Normalization transfer deadline")


def run_bounded(command, log, bounds):
    bounds.check()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "PYTHONPATH":"src:.", "CUDA_VISIBLE_DEVICES":"", "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"})
        try:
            while proc.poll() is None:
                bounds.check()
                time.sleep(.2)
            bounds.check()
            if proc.returncode:
                raise RuntimeError("CPU child failed; inspect " + str(log))
        finally:
            frozen.parent.terminate(proc)


def finished(folder, name):
    return all((folder / (name + suffix)).is_file() for suffix in (".json", "_scores.npz", "_bootstrap.npz"))


def validate_finished(folder, entry, phase, raw_sha256):
    receipt = read(folder / (entry["name"] + ".json"))
    if not receipt or any(receipt.get(k) != v for k, v in dict(status="complete", dataset=entry["name"], phase=phase,
            raw_sha256=raw_sha256, weights_sha256=WEIGHTS_SHA, source_sha256=SOURCE_SHA,
            plan_sha256=sha(OUTPUT / "PLAN.json"), transparent_proxy_original_byte_exact=True, target_fit=False,
            bootstrap_samples=COUNT).items()):
        raise ValueError("Stale or incomplete normalization-transfer receipt")
    for suffix, key in (("_scores.npz", "scores_sha256"), ("_bootstrap.npz", "bootstrap_sha256")):
        if sha(folder / (entry["name"] + suffix)) != receipt.get(key):
            raise ValueError("Normalization-transfer prediction/bootstrap changed")
    expected = ["original_512"] if phase == "source" else list(CASES)
    if [c["case"] for c in receipt["frozen_cases"]] != expected or not all(c["nll_byte_exact"] for c in receipt["frozen_cases"]):
        raise ValueError("Missing declared frozen-control cases")
    if phase == "source" and receipt.get("source_lp_only_saved_prediction_parity") is not True:
        raise ValueError("Source LP-only byte parity not certified")


def run_workers(plan, phase, indices, folder, bounds):
    def one(i):
        name = plan["datasets"][i]["name"]
        if finished(folder, name):
            e = plan["datasets"][i]
            raw_sha = e["source_raw_sha256"] if phase == "source" else next(x["raw_sha256"] for x in read(folder.parent / "cache_audit.json")["per_dataset"] if x["dataset"] == name)
            validate_finished(folder, e, phase, raw_sha)
            return
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-phase", phase, "--dataset-index", str(i), "--worker-output", rel(folder)]
        run_bounded(command, folder / (name + ".log"), bounds)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, i) for i in indices]
        try:
            for f in concurrent.futures.as_completed(futures):
                f.result()
        except BaseException:
            bounds.stop.set()
            raise


def interval(values):
    import numpy as np
    finite = values[np.isfinite(values)]
    return dict(usable_draws=len(finite), ci95_pp=np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None)


def metric_summary(points, boots, names):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    if not all(name in points and name in boots for name in names):
        return dict(status="partial_no_macro", expected_datasets=len(names), present_datasets=len(set(points) & set(names)))
    output = dict(status="complete", datasets=len(names), scopes={}, per_dataset={})
    for scope in ("full", "strict"):
        data, draws = {}, {}
        for method in METHODS:
            a = []
            for name in names:
                z = points[name];mask = np.ones(len(z["labels"]), dtype=bool) if scope == "full" else z["duplicate_disjoint_mask"]
                a.append(100 * roc_auc_score(z["labels"][mask], z[method][mask]))
            data[method] = np.asarray(a)
            draws[method] = 100 * np.stack([boots[name][scope + "/" + method] for name in names])
        for i, name in enumerate(names):
            output["per_dataset"].setdefault(name, {})[scope] = {m:dict(directed_auc=float(data[m][i]), folded_auc=float(max(data[m][i],100-data[m][i]))) for m in METHODS}
        valid = np.all(np.isfinite(np.stack(list(draws.values()))), axis=(0, 1))
        result = {}
        for style in ("directed", "folded"):
            transform = (lambda a:a) if style == "directed" else (lambda a:np.maximum(a, 100-a))
            p = {m:transform(a) for m,a in data.items()}
            b = {m:transform(a) for m,a in draws.items()}
            result[style] = {m:dict(macro_auc=float(np.mean(p[m])), paired_claim_interval=interval(np.mean(b[m][:, valid], axis=0))) for m in METHODS}
            for label, a, bkey in (("fixed_minus_original", FIXED, ORIGINAL), ("fixed_minus_nll", FIXED, NLL), ("original_minus_nll", ORIGINAL, NLL)):
                result[style][label] = dict(delta_pp=float(np.mean(p[a]-p[bkey])), paired_claim_interval=interval(np.mean((b[a]-b[bkey])[:,valid], axis=0)))
        output["scopes"][scope] = result
    return output


def external_parity(plan, points, boots, names, coverage):
    import numpy as np
    rows = []
    for name in names:
        entry = next(e for e in plan["datasets"] if e["name"] == name)
        paths = [local(entry["six_nll_reference"])] if entry["historical_six"] else []
        for protocol in sorted((NIGHT / "transfer_llama32_1b").glob("*/transfer_protocol.json")):
            p = read(protocol)
            audit = (p or {}).get("target_cache_audit") or {}
            if p and p.get("coverage") == "nine" and set(p.get("prespecified_transfer_subset", [])) == {e["name"] for e in plan["datasets"]} and audit.get("status") == "passed" and audit.get("numerical_reference_preflight_verified") is True:
                paths += [protocol.parent / sub / (name + "_scores.npz") for sub in ("new", "supervised")]
        matches = []
        for path in paths:
            if not path.is_file():
                continue
            try:
                z = load_npz(path);b = load_npz(path.with_name(name + "_bootstrap.npz"))
            except (OSError, ValueError, EOFError):
                continue
            if provenance.BASELINE not in z or provenance.BASELINE not in b:
                continue
            require_rows(z, points[name])
            if not frozen.parent.exact(z[provenance.BASELINE], points[name][NLL]) or not np.array_equal(b[provenance.BASELINE], boots[name]["full/" + NLL], equal_nan=True):
                raise ValueError("External target NLL predictions/bootstrap mismatch: " + name)
            if frozen.parent.METHOD in z and not frozen.parent.exact(z[frozen.parent.METHOD], points[name][ORIGINAL]):
                raise ValueError("External original C8741 target mismatch")
            matches.append(dict(path=rel(path), sha256=sha(path), nll_and_rows_exact=True, nll_bootstrap_exact=True))
        rows.append(dict(dataset=name, status="exact_match" if matches else "pending_main_target_reference", references=matches))
    return dict(status="exact_match" if all(x["status"] == "exact_match" for x in rows) else "pending_main_target_reference", per_dataset=rows)


def summarize(plan, coverage):
    import numpy as np
    assert_protected(plan)
    if coverage == "six" and (OUTPUT / "SIX_IMMUTABLE.json").exists():
        assert_six_immutable()
        return read(OUTPUT / "six/TRANSFER_REPORT.json")
    if coverage == "nine":
        receipt = assert_six_immutable()
        for e in plan["datasets"]:
            if e["historical_six"]:
                for suffix in ("_scores.npz", "_bootstrap.npz", ".json"):
                    original = OUTPUT / "six/target" / (e["name"] + suffix)
                    copied = OUTPUT / "nine/target" / original.name
                    if not copied.is_file() or sha(copied) != receipt["files"][rel(original)]:
                        raise ValueError("Copied historical-six result changed")
    entries = plan["datasets"] if coverage == "nine" else [e for e in plan["datasets"] if e["historical_six"]]
    names = [e["name"] for e in entries]
    cohort = OUTPUT / coverage
    audit = read(cohort / "cache_audit.json")
    if not audit or audit.get("status") != "passed" or set(e["dataset"] for e in audit["per_dataset"]) != set(names):
        raise ValueError("Complete prescribed target cache audit required")
    if coverage == "nine":
        completion = read(local(plan["completion_receipt"]))
        if not completion or not provenance.completion_numerical_reference_passed(completion) or audit.get("numerical_reference_preflight_verified") is not True or audit.get("completion_receipt_sha256") != sha(local(plan["completion_receipt"])):
            raise ValueError("Nine-dataset numeric/completion provenance changed")
    data, boots = {}, {}
    for view, folder in (("source", OUTPUT / "source"), ("target", cohort / "target")):
        data[view], boots[view] = {}, {}
        for e in entries:
            if not finished(folder, e["name"]):
                continue
            expected_raw = e["source_raw_sha256"] if view == "source" else next(x["raw_sha256"] for x in audit["per_dataset"] if x["dataset"] == e["name"])
            validate_finished(folder, e, view, expected_raw)
            z = load_npz(folder / (e["name"] + "_scores.npz"));b = load_npz(folder / (e["name"] + "_bootstrap.npz"))
            expected = load_npz(local(e["source_scores"]))
            require_rows(z, expected)
            if view == "source" and any(not frozen.parent.exact(z[m], expected[k]) for m,k in ((ORIGINAL,"original_primary512"),(FIXED,"original_512"),(NLL,"canonical_nll"))):
                raise ValueError("Saved source parity changed")
            if any(z[m].shape != z["labels"].shape or not np.isfinite(z[m]).all() for m in METHODS):
                raise ValueError("Incomplete/nonfinite score array")
            if any(b.get(scope + "/" + m, np.array([])).shape != (COUNT,) for scope in ("full", "strict") for m in METHODS):
                raise ValueError("Missing 500 paired claim draws")
            data[view][e["name"]], boots[view][e["name"]] = z, b
    complete = all(len(data[x]) == len(names) for x in data)
    parity = external_parity(plan, data["target"], boots["target"], list(data["target"]), coverage) if data["target"] else dict(status="pending_main_target_reference")
    ready = complete and parity["status"] == "exact_match"
    result = dict(status="complete_exploratory_" + coverage if ready else "partial_or_external_reference_pending",
                  coverage=coverage, expected_datasets=names, source_fitting=False, target_fitting=False, primary_replaced=False,
                  weights_sha256=WEIGHTS_SHA, plan_sha256=sha(OUTPUT / "PLAN.json"), external_nll_parity=parity,
                  source=metric_summary(data["source"], boots["source"], names), target=metric_summary(data["target"], boots["target"], names) if ready else dict(status="pending_no_macro"),
                  matched_transfer={}, limits=plan["scientific_limits"])
    if ready:
        for scope in ("full", "strict"):
            result["matched_transfer"][scope] = {}
            for style in ("directed", "folded"):
                transform = (lambda a:a) if style == "directed" else (lambda a:np.maximum(a, 100-a))
                deltas = {}
                for method in METHODS:
                    sb = 100 * np.stack([boots["source"][n][scope + "/" + method] for n in names])
                    tb = 100 * np.stack([boots["target"][n][scope + "/" + method] for n in names])
                    deltas[method] = transform(tb)-transform(sb)
                    point = result["target"]["scopes"][scope][style][method]["macro_auc"]-result["source"]["scopes"][scope][style][method]["macro_auc"]
                    valid = np.all(np.isfinite(deltas[method]), axis=0)
                    result["matched_transfer"][scope][style + "/" + method] = dict(target_minus_source_pp=point, paired_claim_interval=interval(np.mean(deltas[method][:, valid], axis=0)))
                gain = deltas[FIXED]-deltas[NLL];valid = np.all(np.isfinite(gain),axis=0)
                point = result["target"]["scopes"][scope][style]["fixed_minus_nll"]["delta_pp"]-result["source"]["scopes"][scope][style]["fixed_minus_nll"]["delta_pp"]
                result["matched_transfer"][scope][style + "/fixed_nll_gain_change"] = dict(target_minus_source_pp=point, paired_claim_interval=interval(np.mean(gain[:,valid],axis=0)))
    cohort.mkdir(parents=True, exist_ok=True)
    dump(cohort / "TRANSFER_REPORT.json", result)
    lines = ["# Source-fixed C8741 normalization transfer", "", "Status: **" + result["status"] + "**.", "",
             "Three frozen detectors: original batch-normalized C8741 (512 rows), source-fitted fixed normalization, canonical NLL. No target fitting or selection. Source and target use the same prescribed " + str(len(names)) + " datasets. Six results are preserved when three datasets are added.", "",
             "| Model / scope | Original directed / folded | Fixed directed / folded | NLL directed / folded |", "| --- | ---: | ---: | ---: |"]
    for view in ("source", "target"):
        summary = result[view]
        if summary["status"] != "complete":
            continue
        for scope in ("full", "strict"):
            s = summary["scopes"][scope]
            cells = [f"{s['directed'][m]['macro_auc']:.4f} / {s['folded'][m]['macro_auc']:.4f}" for m in METHODS]
            lines.append("| " + view + " / " + scope + " | " + " | ".join(cells) + " |")
    lines += ["", "500 deterministic paired claim draws; strict draws use only retained claim groups. Per-dataset folding precedes each fixed-dataset mean. Full paired intervals and matched-transfer differences are in JSON.", "",
              "Source zero-MAD rules permanently disable f3/f4 on target. This tests a frozen feature/scale combination; medians shift an offset and AUROC does not establish calibration transfer. Exploratory after source benchmark viewing; no original primary is replaced."]
    (cohort / "TRANSFER_REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def execute(plan, coverage, wait=False):
    if ROOT != Path("/workspace/mod1"):
        raise ValueError("Actual cached scoring restricted to authorized pod")
    bounds = Bounds(DEADLINE)
    bounds.check()
    import fcntl
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "execution.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if coverage == "nine":
            reuse_six(plan)
            while True:
                bounds.check()
                completion = read(local(plan["completion_receipt"]))
                if completion and completion.get("status") == "complete_three_missing_caches":
                    if not provenance.completion_numerical_reference_passed(completion):
                        raise ValueError("Formal completion lacks passed 427-row numerical guard")
                    break
                if not wait:
                    return dict(status="waiting_for_formal_nine_numeric_guard")
                time.sleep(min(15, max(0, (DEADLINE-now()).total_seconds())))
        history = read(OUTPUT / "execution_history.json") or dict(active_seconds=0., attempts=[])
        remaining = MAX_SECONDS-history["active_seconds"]
        if remaining < 30:
            raise TimeoutError("Cumulative 20-minute CPU budget exhausted")
        start = time.monotonic();bounds.deadline = min(DEADLINE, now()+dt.timedelta(seconds=remaining))
        attempt = dict(coverage=coverage, started_utc=now().isoformat(), deadline_utc=bounds.deadline.isoformat(), status="running")
        history["attempts"].append(attempt);dump(OUTPUT / "execution_history.json", history)
        def interrupted(signum, frame):
            bounds.stop.set()
            raise KeyboardInterrupt("Normalization transfer interrupted")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            # Audit occurs as a bounded CPU child, not an unbounded parent call.
            cohort = OUTPUT / coverage
            run_bounded([sys.executable, str(Path(__file__).resolve()), "--audit-coverage", coverage], cohort / "cache_audit.log", bounds)
            run_workers(plan, "source", list(range(9)), OUTPUT / "source", bounds)
            if not all(finished(OUTPUT / "source", e["name"]) for e in plan["datasets"]):
                raise ValueError("All-nine LP-only source parity required before target scoring")
            indices = [e["index"] for e in plan["datasets"] if e["historical_six"] == (coverage == "six")]
            run_workers(plan, "target", indices, cohort / "target", bounds)
            result = summarize(plan, coverage)
            bounds.check()
            if coverage == "six" and result["status"] == "complete_exploratory_six":
                freeze_six()
            if coverage == "nine":
                reuse_six(plan)
            attempt["status"] = result["status"]
            return result
        except BaseException as exc:
            bounds.stop.set();attempt.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            history["active_seconds"] += time.monotonic()-start
            attempt["finished_utc"] = now().isoformat()
            dump(OUTPUT / "execution_history.json", history)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true");ap.add_argument("--coverage", choices=["six", "nine"], default="six")
    ap.add_argument("--wait", action="store_true");ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--audit-coverage", choices=["six", "nine"])
    ap.add_argument("--worker-phase", choices=["source", "target"]);ap.add_argument("--dataset-index", type=int)
    ap.add_argument("--worker-output");args = ap.parse_args()
    os.chdir(ROOT)
    if args.worker_phase or args.audit_coverage or args.report_only:
        plan = read(OUTPUT / "PLAN.json");assert_protected(plan)
        if args.worker_phase:
            worker(plan, args.dataset_index, args.worker_phase, local(args.worker_output))
        elif args.audit_coverage:
            target_audit(plan, args.audit_coverage, OUTPUT / args.audit_coverage)
        else:
            print(summarize(plan, args.coverage)["status"])
        return
    plan = prepare()
    if not args.execute:
        print("Prepared only; no cached scoring, fitting, generation or GPU calls.")
        return
    print(execute(plan, args.coverage, args.wait)["status"])


if __name__ == "__main__":
    main()
