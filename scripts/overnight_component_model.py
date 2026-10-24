"""Guarded frozen inference for the exploratory screen-component logistic head.

Components are the actual frozen full-program output minus its corresponding
drop-program output, including original precision and branch behaviour.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def guarded_path(root, relative, expected):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or sha(path) != expected:
        raise ValueError("Frozen component input path/hash mismatch: " + str(relative))
    return path


def load_detector(path, context):
    from experiments.extension_trial import parse_candidate
    from packing.evaluate.uncertainty_quantification.estimator_base import Estimator
    code, imports = parse_candidate(Path(path).read_text())
    namespace = dict(Estimator=Estimator, np=np, scipy=scipy, Any=Any, Dict=Dict,
                     List=List, Tuple=Tuple, Optional=Optional)
    exec(imports + "\n" + code, namespace)
    return namespace["HallucinationDetector"](context)


class ComponentPool:
    def __init__(self, root, pool_relative, pool_sha256, context=None):
        self.root = Path(root)
        self.manifest = json.loads(guarded_path(root, pool_relative, pool_sha256).read_text())
        context = context or {"supervised": False, "meta": {"eval_meta": {}}}
        self.detectors = {}
        for item in self.manifest["programs"]:
            path = guarded_path(root, item["file"], item["sha256"])
            self.detectors[item["method"]] = load_detector(path, context)
        self.feature_names = [item["name"] for item in self.manifest["components"]] + ["float64_total_nll"]
        self.dependencies = sorted({key for detector in self.detectors.values() for key in detector.stats_dependencies})
        if len(self.manifest["components"]) != 11 or len(self.detectors) != 15:
            raise ValueError("Expected all four full programs and all eleven component drops")

    def transform(self, stats):
        n = len(stats["sequence_logprobs"])
        outputs = {}
        for key, detector in self.detectors.items():
            private = copy.deepcopy({name: stats[name] for name in detector.stats_dependencies})
            result = np.asarray(detector(private), dtype=np.float64)
            if result.shape != (n,) or not np.isfinite(result).all():
                raise ValueError("Invalid frozen component-program output: " + key)
            outputs[key] = result
        columns = [outputs[item["full"]] - outputs[item["drop"]] for item in self.manifest["components"]]
        # An unconditional predefined reference is retained even though two
        # branch-dependent LLM components resemble NLL. No rank/quality pruning.
        columns.append(np.asarray([-np.sum(np.asarray(lp, dtype=np.float64))
                                   for lp in stats["sequence_logprobs"]]))
        result = np.column_stack(columns)
        if result.shape != (n, 12) or not np.isfinite(result).all():
            raise ValueError("Nonfinite component feature matrix")
        return result


class FrozenComponentLogistic:
    def __init__(self, context, artifact_relative, artifact_sha256, module_sha256):
        if sha(__file__) != module_sha256:
            raise ValueError("Frozen component inference module hash mismatch")
        artifact = json.loads(guarded_path(ROOT, artifact_relative, artifact_sha256).read_text())
        if artifact["inference_module_sha256"] != module_sha256:
            raise ValueError("Artifact/module provenance mismatch")
        self.pool = ComponentPool(ROOT, artifact["pool_file"], artifact["pool_sha256"], context)
        if self.pool.feature_names != artifact["feature_names"]:
            raise ValueError("Frozen feature ordering mismatch")
        self.mean = np.asarray(artifact["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(artifact["scaler_scale"], dtype=np.float64)
        self.coefficients = np.asarray(artifact["coefficients_standardized"], dtype=np.float64)
        self.intercept = float(artifact["intercept"])
        if any(x.shape != (12,) or not np.isfinite(x).all() for x in [self.mean, self.scale, self.coefficients]):
            raise ValueError("Invalid frozen coefficient dimensions/values")
        if np.any(self.scale <= 0) or not np.isfinite(self.intercept):
            raise ValueError("Invalid frozen scaler/intercept")
        self.stats_dependencies = self.pool.dependencies

    def __call__(self, stats):
        features = self.pool.transform(stats)
        return ((features - self.mean) / self.scale) @ self.coefficients + self.intercept
