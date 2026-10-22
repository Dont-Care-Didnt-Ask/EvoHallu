"""Scientific invariants for feature-extractor transfer; no model downloads or GPU."""
import csv
import hashlib
import importlib.util
import io
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from experiments.feature_transfer import cache, config, data, evaluate, methods, report
from experiments.feature_transfer.features import features, render
from experiments.feature_transfer.io import sha256


class TransferProtocolTests(unittest.TestCase):
    def test_claim_normalization_preserves_the_recorded_operation_order(self):
        self.assertEqual(data.normalize(" \r\nПривет\rмир\n "), "Привет\nмир")
        self.assertEqual(data.normalize("🙂" * 301), "🙂" * 300)
        # Truncation happens after strip, so an internal space can become the last character.
        self.assertEqual(data.normalize("x" * 299 + " y"), "x" * 299 + " ")

    def test_zip_member_mapping_does_not_depend_on_archive_or_request_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "sources.zip"
            pins = {}
            with ZipFile(archive_path, "w") as archive:
                for key in ("counterfact", "commonclaim"):
                    stream = io.StringIO(newline="")
                    writer = csv.writer(stream)
                    writer.writerow(["claim", "is_hallu"])
                    writer.writerows([(f"{key} first", 0), (f"{key} second", 1)])
                    payload = stream.getvalue().encode()
                    member = config.DATASETS[key]["member"]
                    archive.writestr(member, payload)
                    pins[key] = {"member": member, "sha256": hashlib.sha256(payload).hexdigest(),
                                 "rows": 2, "is_hallu_1": 1}
            with patch.object(data, "DATASETS", pins), patch.object(data, "ARCHIVE_SHA256", sha256(archive_path)):
                datasets = data.load_datasets(archive_path, ["commonclaim", "counterfact"])
                self.assertEqual(datasets["commonclaim"]["rows"][0]["claim"], "commonclaim first")
                self.assertEqual(datasets["counterfact"]["rows"][0]["row_id"], "counterfact:0")
                pins["commonclaim"]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "Source checksum"):
                    data.load_datasets(archive_path, ["commonclaim"])

    def test_frozen_methods_are_available_and_reject_source_changes(self):
        loaded = methods.load_methods()
        self.assertEqual(len(loaded), 25)
        self.assertEqual(loaded[0][0], "SequenceLogProb")
        self.assertEqual(tuple(name for name, _ in loaded), config.METHOD_NAMES)
        self.assertEqual(methods.SOURCE_HASHES, {
            "found_formulas": "ac1ed0b7fa2884c9f326a861f68777db75fba7b18bcd8ef565b78fa225dabebc",
            "estimator_base": "67f791cf66e84968e318aa6b113f043eba268785dc198f8d9c4aaa023d8c6dd5",
        })
        with patch.object(methods, "sha256", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "formulas differ"):
                methods.load_methods()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch is needed for numerical feature checks")
    def test_claim_features_use_the_previous_token_predictor(self):
        import torch
        ids = torch.tensor([7, 9, 11, 13, 15])
        logits = torch.zeros((5, 512), dtype=torch.float32)
        for position in range(4):
            logits[position, ids[position + 1]] = position + 1
        result = features(logits, ids, [2, 4])
        target_logits = np.array([2.0, 4.0])
        expected = target_logits - np.log(np.exp(target_logits) + 511)
        np.testing.assert_allclose(result["claim_sequence_logprobs"], expected, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(result["claim_sequence_logprobs"], result["sequence_logprobs"][[1, 3]])
        np.testing.assert_array_equal(result["claim_token_entropies"], result["token_entropies"][[1, 3]])
        np.testing.assert_array_equal(result["claim_top_k_logits"], result["top_k_logits"][[1, 3]])
        np.testing.assert_array_equal(result["claim_greedy_logprobs"][1:], result["greedy_logprobs"][[1, 3]])
        self.assertEqual(result["claim_greedy_logprobs"][0], 0)
        self.assertEqual(result["top_k_logits"].dtype, np.float16)

    def test_render_rejects_a_token_crossing_the_claim_boundary(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **options):
                self.claim = messages[-1]["content"]
                self.options = options
                return "prefix|" + self.claim + "|suffix"

            def __call__(self, text, **options):
                start = text.index(self.claim)
                offsets = [(0, 0)] + [(i, i + 1) for i in range(len(text))]
                offsets[start + 1] = (start - 1, start + 1)
                return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

        tokenizer = Tokenizer()
        with self.assertRaisesRegex(ValueError, "boundaries"):
            render(tokenizer, "claim", "gemma")
        self.assertFalse(tokenizer.options["enable_thinking"])
        self.assertFalse(tokenizer.options["add_generation_prompt"])

    @staticmethod
    def _sample(index):
        entropy = np.asarray([0.1, 0.2, 0.3, 0.5, 0.9], dtype=np.float32)
        if index:
            entropy = entropy[::-1].copy()
        top = np.tile(np.linspace(1, -1, 512, dtype=np.float16), (5, 1))
        return {
            "sequence_logprobs": np.asarray([-1, -2, -3, -4], dtype=np.float32),
            "greedy_logprobs": np.asarray([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=np.float32),
            "token_entropies": entropy, "top_k_logits": top,
            "input_ids": np.arange(5, dtype=np.int32),
            "top_k_indices": np.tile(np.arange(512, dtype=np.int32), (5, 1)),
            "claim_indices": np.asarray([2, 4], dtype=np.int32),
            "claim_sequence_logprobs": np.asarray([-2, -4], dtype=np.float32),
            "claim_greedy_logprobs": np.asarray([0, -0.2, -0.4], dtype=np.float32),
            "claim_token_entropies": entropy[[1, 3]], "claim_top_k_logits": top[[1, 3]],
        }

    def _cache(self, directory):
        rows = [{"row_id": f"cities:{i}", "label": i} for i in range(2)]
        dataset = {"rows": rows, "pin": config.DATASETS["cities"]}
        settings = {"dtype": "bf16", "fixture": True}
        stored = cache.DatasetCache(directory, "llama", "cities", dataset, {}, settings)
        for i in range(2):
            stored.commit(cache.write_shard(directory, rows[i:i + 1], [self._sample(i)], i, {}))
        return stored, dataset, settings

    def test_sharded_cache_roundtrip_resume_and_corruption_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stored, dataset, settings = self._cache(directory)
            for view in ("full", "claim"):
                manifest, ids, labels, stats = evaluate.read_cache(stored.path, view)
                self.assertFalse(manifest["full_dataset"])
                self.assertEqual(ids.tolist(), ["cities:0", "cities:1"])
                self.assertEqual(labels.tolist(), [0, 1])
                prefix = "claim_" if view == "claim" else ""
                for i in range(2):
                    for field in config.FIELDS:
                        np.testing.assert_array_equal(stats[field][i], self._sample(i)[prefix + field])
            self.assertEqual(cache.DatasetCache(directory, "llama", "cities", dataset, {}, settings).cursor, 2)
            with self.assertRaisesRegex(ValueError, "source order"):
                stored.commit([stored.manifest["shards"][0], stored.manifest["provenance_shards"][0]])
            with self.assertRaisesRegex(ValueError, "settings differ"):
                cache.DatasetCache(directory, "llama", "cities", dataset, {}, {"dtype": "fp32"})
            reordered = {**dataset, "rows": dataset["rows"][::-1]}
            with self.assertRaisesRegex(ValueError, "inconsistent cache"):
                cache.DatasetCache(directory, "llama", "cities", reordered, {}, settings)
            shard = directory / stored.manifest["shards"][0]["path"]
            with shard.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                evaluate.read_cache(stored.path, "full")
            with self.assertRaisesRegex(ValueError, "inconsistent cache"):
                cache.DatasetCache(directory, "llama", "cities", dataset, {}, settings)

    def test_formula_normalization_receives_the_whole_dataset_across_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stored, _, _ = self._cache(directory / "cache")
            estimator = dict(methods.load_methods())["QuartileEntropyDiffNormalized"]
            observed_sizes = []

            def track(stats):
                observed_sizes.append(len(stats["token_entropies"]))
                return estimator(stats)

            with patch.object(evaluate, "load_methods", return_value=[("QuartileEntropyDiffNormalized", track)]):
                pair = evaluate.evaluate_cache(stored.path, directory / "scores")
            self.assertEqual(observed_sizes, [2, 2])
            self.assertEqual(len(pair["metrics"]), 2)
            with np.load(directory / "scores" / pair["score_files"]["full"]["path"]) as scores:
                np.testing.assert_allclose(sorted(scores["QuartileEntropyDiffNormalized"]), [-1, 1], atol=1e-6)

    def test_fold_each_dataset_before_macro_and_withhold_incomplete_average(self):
        metrics = []
        for model, dataset, view, method in product(config.MODELS, config.DATASET_ORDER,
                                                    ("full", "claim"), config.METHOD_NAMES):
            fixed = 0.2 if config.DATASET_ORDER.index(dataset) < 4 else 0.8
            metrics.append({"model": model, "dataset": dataset, "view": view, "method": method,
                            "rows": config.DATASETS[dataset]["rows"], "positive": config.DATASETS[dataset]["is_hallu_1"],
                            "full_dataset": True, "auc_fixed": fixed,
                            "auc_folded_compatibility": max(fixed, 1 - fixed), "status": "passed"})
        macro = {(r["mode"], r["model"], r["method"]): r for r in report.build_macro(metrics)}
        self.assertEqual(len(macro), 225)
        folded = macro["full_folded", "llama", "SequenceLogProb"]["macro_auc"]
        fixed = macro["full_fixed", "llama", "SequenceLogProb"]["macro_auc"]
        self.assertAlmostEqual(folded, 0.8)
        self.assertAlmostEqual(fixed, (4 * 0.2 + 5 * 0.8) / 9)
        self.assertGreater(folded, max(fixed, 1 - fixed))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report.write_report(metrics, directory)
            self.assertTrue((directory / "average.md").exists())
            for incomplete in (metrics[:-1], [{**r, "full_dataset": False} for r in metrics]):
                with self.assertRaisesRegex(ValueError, "Mean9 requires"):
                    report.build_macro(incomplete)
                self.assertEqual(report.write_report(incomplete, directory), [])
                self.assertFalse((directory / "average.md").exists())
                with (directory / "macro9.csv").open(newline="") as stream:
                    self.assertEqual(list(csv.DictReader(stream)), [])

    def test_invalid_rerun_does_not_leave_a_previous_complete_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            directory.joinpath("average.md").write_text("obsolete complete Average")
            directory.joinpath("evaluation.json").write_text('{"status":"complete"}')
            with self.assertRaisesRegex(ValueError, "Expected one complete cache"):
                evaluate.evaluate(directory / "no_caches", directory)
            self.assertFalse(directory.joinpath("average.md").exists())
            self.assertFalse(directory.joinpath("evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
