from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import FIELDS, MODELS
from .io import read_json, sha256, write_json


def pack_ragged(values):
    lengths = np.asarray([len(value) for value in values], dtype=np.int64)
    return np.concatenate(values), np.r_[np.int64(0), np.cumsum(lengths)]


def save_npz(path: Path, arrays, compressed=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        save = np.savez_compressed if compressed else np.savez
        save(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_shard(directory: Path, rows, values, start: int, compute):
    stop = start + len(rows)
    identity = {
        "row_ids": np.asarray([row["row_id"] for row in rows]),
        "labels": np.asarray([row["label"] for row in rows], dtype=np.uint8),
    }
    feature_names = [prefix + field for prefix in ("", "claim_") for field in FIELDS]
    groups = [
        ("stats", feature_names, False),
        ("provenance", ("input_ids", "top_k_indices", "claim_indices"), True),
    ]
    entries = []
    for kind, fields, compressed in groups:
        arrays = dict(identity)
        for field in fields:
            arrays[field + "_values"], arrays[field + "_offsets"] = pack_ragged(
                [value[field] for value in values]
            )
        path = directory / f"{kind}_{start:09d}_{stop:09d}.npz"
        save_npz(path, arrays, compressed)
        entries.append({
            "path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
            "rows": len(rows), "start_row": start, "stop_row": stop,
        })
    entries[0]["compute"] = compute
    return entries


class DatasetCache:
    def __init__(self, directory: Path, model_key, dataset_key, dataset, metadata, settings):
        self.directory = directory
        self.rows = dataset["rows"]
        self.path = directory / "manifest.json"
        directory.mkdir(parents=True, exist_ok=True)
        pin = dataset["pin"]
        model_id, revision = MODELS[model_key]
        identity = {
            "model": (model_id, revision), "dataset_sha256": pin["sha256"],
            "rows": len(self.rows), "settings": settings,
            "template_sha256": metadata.get("template_sha256"),
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if self.path.exists():
            self.manifest = read_json(self.path)
            if self.manifest["run_identity_sha256"] != fingerprint:
                raise ValueError("Resume settings differ; use a new output directory")
            self._verify_resume()
        else:
            self.manifest = {
                "schema": "evohallu.logits_cache.v1", "model_key": model_key,
                "model_id": model_id, "model_revision": revision, "dataset_key": dataset_key,
                "dataset_sha256": pin["sha256"], "archive_member": pin["member"],
                "expected_rows": len(self.rows), "source_rows": pin["rows"],
                "source_positive": pin["is_hallu_1"], "top_k": 512,
                "pilot_only": (len(self.rows) != pin["rows"]
                               or settings.get("dtype", "bf16") != "bf16"
                               or settings.get("pilot_only", False)),
                "status": "running",
                "cached_rows": 0, "run_identity_sha256": fingerprint,
                "shards": [], "provenance_shards": [], "meta": metadata,
            }
            write_json(self.path, self.manifest)

    @property
    def cursor(self):
        return self.manifest["cached_rows"]

    def _verify_resume(self):
        manifest = self.manifest
        stats, tokens = manifest["shards"], manifest["provenance_shards"]
        if len(stats) != len(tokens):
            raise ValueError("Cache stats and token provenance are incomplete")
        cursor = 0
        for stats_entry, token_entry in zip(stats, tokens):
            stop = cursor + stats_entry["rows"]
            rows = self.rows[cursor:stop]
            for entry in (stats_entry, token_entry):
                path = self.directory / entry["path"]
                valid = (entry["start_row"] == cursor and entry["stop_row"] == stop
                         and entry["rows"] == len(rows) and sha256(path) == entry["sha256"])
                with np.load(path, allow_pickle=False) as shard:
                    valid &= (shard["row_ids"].tolist() == [row["row_id"] for row in rows]
                              and shard["labels"].tolist() == [row["label"] for row in rows])
                if not valid:
                    raise ValueError(f"Cannot resume inconsistent cache: {path}")
            cursor = stop
        if cursor != manifest["cached_rows"] or cursor > len(self.rows):
            raise ValueError("Cache progress does not match committed shards")

    def commit(self, entries):
        stats, tokens = entries
        if stats["start_row"] != self.cursor:
            raise ValueError("Shards must commit in source order")
        self.manifest["shards"].append(stats)
        self.manifest["provenance_shards"].append(tokens)
        self.manifest["cached_rows"] = stats["stop_row"]
        self.manifest["status"] = "complete" if self.cursor == len(self.rows) else "running"
        write_json(self.path, self.manifest)
