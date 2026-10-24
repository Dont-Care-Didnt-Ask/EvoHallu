import json
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from pathlib import Path

from experiments.extension_trial import parse_candidate, build_prompt
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import generate_input
from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import _prepare_train_val_indices


def test_method_local_imports_are_not_hoisted():
    code, imports = parse_candidate('''Some reasoning.\n```python
import numpy as np
class HallucinationDetector(Estimator):
    def __call__(self, stats):
        import scipy
        return np.zeros(len(stats["sequence_logprobs"]))
```''')
    assert imports == "import numpy as np"
    assert "        import scipy" in code


def test_training_and_validation_never_open_test_file(tmp_path):
    samples = [{"label": i % 2} for i in range(20)]
    raw = tmp_path / "train.pt"
    torch.save({"samples": samples}, raw)
    cfg = OmegaConf.create(dict(train_raw_path=str(raw), test_raw_path=str(tmp_path / "missing_test.pt"),
                                sample_size=0, val_size=0.2, split_seed=441, supervised=False))
    assert len(generate_input(cfg, "train")["eval_raw"]["samples"]) == 16
    assert len(generate_input(cfg, "val")["eval_raw"]["samples"]) == 4
    with pytest.raises(FileNotFoundError):
        generate_input(cfg, "testset")


def test_independent_prompt_never_samples_population():
    class ForbiddenDatabase:
        def get_prompt(self, **kwargs):
            raise AssertionError("independent sampling accessed evolved population")
    cfg = OmegaConf.create({"search_mode": "independent", "seed": 1046})
    chat, island = build_prompt(cfg, ForbiddenDatabase(), "SECRET_BASELINE_CODE", 99, 0)
    assert island is None
    assert all("SECRET_BASELINE_CODE" not in m["content"] for m in chat)
    assert "def __call__(self, stats)" in chat[-1]["content"]
    assert "def __init__(self, context)" in chat[-1]["content"]


def test_trial_config_composes():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="config_uq_trial")
    assert cfg.model.hf_model_id == "Qwen/Qwen3.8-27B"
    assert cfg.trial.local_only
    assert cfg.task.bootstrap_samples == 0


def test_group_split_keeps_repeated_claims_together():
    samples = [{"claim": f"claim {i // 2}", "label": (i // 2) % 2} for i in range(100)]
    train, val = _prepare_train_val_indices({"samples": samples}, 0, 0.2, 441, "claim_group")
    assert not ({samples[i]["claim"] for i in train} & {samples[i]["claim"] for i in val})
    assert set(train) | set(val) == set(range(100))


@pytest.mark.parametrize("outputs,native_tokenizer,mode,policy,save_train", [
    (1,False,"evolution","original",False),
    (2,False,"evolution","original",False),
    (1,True,"evolution","original",False),
    (2,False,"frozen_context","diverse",True),
])
def test_local_trial_completes_and_freezes_without_test_data(tmp_path, outputs, native_tokenizer, mode, policy, save_train):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from experiments.extension_trial import run_trial

    content = '''```python
from __future__ import annotations
class HallucinationDetector(Estimator):
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")
    @staticmethod
    def _score(x):
        return -np.mean(x)
    def __call__(self, stats) -> List[float]:
        return np.asarray([HallucinationDetector._score(x) for x in stats["sequence_logprobs"]])
```'''
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            if native_tokenizer:
                assert "chat_template_kwargs" not in payload
            else:
                assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            variant = content.replace("class HallucinationDetector(Estimator):",
                                      "class HallucinationDetector(Estimator):\n    variant = " + str(payload["seed"]))
            body = json.dumps({"id": "test", "object": "chat.completion", "created": 0,
                               "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                               "message": {"role": "assistant", "content": variant}}],
                               "usage": {"prompt_tokens": 60, "completion_tokens": 40, "total_tokens": 100}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    samples = []
    for i in range(40):
        samples.append({"label": i % 2, "claim": f"synthetic {i}", "input_ids": torch.arange(6),
                        "orig_input_ids": torch.arange(4), "sequence_logprobs": np.full(5, -1.0-i%2),
                        "greedy_logprobs": np.full(6, -0.5), "token_entropies": np.ones(6),
                        "top_k_logits": torch.zeros(6, 512), "top_k_indices": torch.zeros(6, 512, dtype=torch.long)})
    raw_path = tmp_path / "train.pt"
    torch.save({"samples": samples, "meta": {}}, raw_path)
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="config_uq_trial")
    cfg.task.train_raw_path = str(raw_path)
    cfg.task.test_raw_path = str(tmp_path / "DO_NOT_OPEN.pt")
    cfg.cluster.api_base_url = f"http://127.0.0.1:{server.server_port}/v1"
    cfg.logs_dir = str(tmp_path / "run")
    cfg.num_rounds = 1
    cfg.num_outputs_per_prompt = outputs
    cfg.trial.max_seconds = 120
    cfg.search_mode = mode
    cfg.trial.prompt_policy = policy
    cfg.trial.save_train_predictions = save_train
    cfg.trial.shortlist_size = 3 if save_train else 1
    if mode == "frozen_context":
        from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import system_prompt_fabric
        manifest = tmp_path / "contexts.json"
        chat = [{"role": "system", "content": system_prompt_fabric()}, {"role": "user", "content": "Fixed training context"}]
        manifest.write_text(json.dumps({"contexts": [{"chat": chat}]}))
        with open_dict(cfg.trial):
            cfg.trial.frozen_contexts_path = str(manifest)
    with open_dict(cfg.model):
        cfg.model.supports_chat_template_kwargs = not native_tokenizer
    try:
        run_trial(cfg)
    finally:
        server.shutdown()
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["requests"] == outputs
    assert summary["valid_candidates"] == outputs
    assert summary["total_tokens"] == outputs * 100
    assert summary["held_out_test_evaluated"] is False
    assert (tmp_path / "run/frozen_estimator.py").is_file()
    actual_prompts = json.loads((tmp_path / "run/candidate_prompts_0000.json").read_text())
    assert {json.dumps(c, sort_keys=True) for c in actual_prompts} == {json.dumps(p["messages"], sort_keys=True) for p in received}
    if mode == "frozen_context":
        assert actual_prompts[0] != actual_prompts[1]
    if save_train:
        assert len(summary["validation_shortlist"]) == 3
        assert summary["validation_shortlist"][0]["candidate"] == summary["selected_candidate"]
        assert all(Path(item["file"]).is_file() for item in summary["validation_shortlist"])
        assert np.load(tmp_path / "run/baseline_train_scores.npy").shape == (32,)
        records = [json.loads(line) for line in (tmp_path / "run/candidates.jsonl").read_text().splitlines()]
        assert all(np.load(row["train_scores_file"]).shape == (32,) for row in records)
        cfg.trial.resume = True
        cfg.trial.prompt_policy = "neutral"
        with pytest.raises(ValueError, match="changed trial.prompt_policy"):
            run_trial(cfg)


def test_weighted_bootstrap_auc_matches_explicit_resampling_with_ties():
    from scripts.evaluate_cached_claims import weighted_auc_bootstrap, bootstrap_counts
    from sklearn.metrics import roc_auc_score
    y = np.array([0, 1, 0, 1, 1])
    pred = np.array([0.2, 0.2, 0.7, 0.8, 0.8])
    counts = np.array([2, 3, 1, 2, 4])
    index = np.repeat(np.arange(len(y)), counts)
    assert weighted_auc_bootstrap(y, pred, counts) == pytest.approx(roc_auc_score(y[index], pred[index]))
    class FixedGroups:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 3, 3)
            return np.array([0, 2, 2])
    clustered = bootstrap_counts(np.array([0, 0, 1, 2, 2]), FixedGroups())
    np.testing.assert_array_equal(clustered, [1, 1, 0, 2, 2])
    explicit = np.repeat(np.arange(len(y)), clustered)
    assert weighted_auc_bootstrap(y, pred, clustered) == pytest.approx(roc_auc_score(y[explicit], pred[explicit]))


def test_logits_only_precompute_preserves_numerical_features(monkeypatch):
    import pandas as pd
    from transformers import LlamaConfig, LlamaForCausalLM, BatchEncoding
    from packing.evaluate.uncertainty_quantification.precompute_raw_stats_separated import _precompute_split
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            rows = [[3 + ord(c) % 100 for c in text] for text in texts]
            width = max(map(len, rows))
            ids = torch.tensor([r + [0] * (width-len(r)) for r in rows])
            mask = torch.tensor([[1] * len(r) + [0] * (width-len(r)) for r in rows])
            return BatchEncoding({"input_ids": ids, "attention_mask": mask})
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=600, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    df = pd.DataFrame({"claim": ["abcd", "defghi"], "is_hallu": [0, 1]})
    common = dict(df=df, tokenizer=Tokenizer(), model=model, text_col="claim", label_col="is_hallu",
                  id_col=None, batch_size=2, chat_template=None, top_k=512, layer_stride=1)
    original = _precompute_split(**common, modalities="all")
    reduced = _precompute_split(**common, modalities="logits")
    observed = []
    hook = model.register_forward_pre_hook(lambda module, args, kwargs: observed.append(kwargs["input_ids"].numel()), with_kwargs=True)
    adaptive = _precompute_split(**common, modalities="logits", max_batch_tokens=6)
    hook.remove()
    assert observed == [4, 6]
    assert [s["claim"] for s in adaptive["logits_struct"]["samples"]] == df.claim.tolist()
    assert [s["label"] for s in adaptive["logits_struct"]["samples"]] == [0., 1.]
    assert reduced["hidden_states_struct"]["samples"] == []
    assert reduced["attentions_struct"]["samples"] == []
    for first, second in zip(original["logits_struct"]["samples"], reduced["logits_struct"]["samples"]):
        for key in ["sequence_logprobs", "greedy_logprobs", "token_entropies", "top_k_logits", "top_k_indices"]:
            np.testing.assert_allclose(np.asarray(first[key]), np.asarray(second[key]), rtol=1e-5, atol=1e-5)
    for first, second in zip(reduced["logits_struct"]["samples"], adaptive["logits_struct"]["samples"]):
        for key in ["sequence_logprobs", "greedy_logprobs", "token_entropies", "top_k_logits", "top_k_indices"]:
            np.testing.assert_allclose(np.asarray(first[key]), np.asarray(second[key]), rtol=1e-5, atol=1e-5)


def test_claim_crop_handles_a_multi_token_template_suffix():
    from packing.evaluate.uncertainty_quantification.alignment_baselines import claim_likelihoods
    full = [100, 101, 8, 9, 10, 200, 201, 202]
    lp = np.array([-.1, -1., -2., -3., -.4, 0., 0.])
    np.testing.assert_array_equal(claim_likelihoods(lp, [8, 9, 10], full), [-1., -2., -3.])


def test_frozen_benchmark_resets_state_and_isolates_inputs(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.evaluate_cached_claims import main
    monkeypatch.chdir(tmp_path)
    raw_samples = [{"label": i % 2, "claim": f"evaluation claim {i}",
                    "input_ids": torch.arange(6), "orig_input_ids": torch.arange(4),
                    "sequence_logprobs": np.full(5, -1.-i%2, dtype=np.float32),
                    "greedy_logprobs": np.full(6, -.5, dtype=np.float32),
                    "token_entropies": np.ones(6, dtype=np.float32),
                    "top_k_logits": torch.zeros(6, 512), "top_k_indices": torch.zeros(6, 512, dtype=torch.long)}
                   for i in range(20)]
    data_root, cache, output = tmp_path / "claims", tmp_path / "cache", tmp_path / "output"
    cache.mkdir()
    (data_root / "popqa").mkdir(parents=True)
    pd.DataFrame({"claim": ["training-only claim"], "is_hallu": [0]}).to_csv(
        data_root / "popqa/popqa_train_filtered_10shot.csv", index=False)
    for i in range(9):
        folder = data_root / f"d{i}"
        folder.mkdir()
        pd.DataFrame({"claim": [s["claim"] for s in raw_samples], "is_hallu": [s["label"] for s in raw_samples]}).to_csv(folder / "test.csv", index=False)
        torch.save({"samples": raw_samples, "meta": {"vocab_size": 600}}, cache / f"d{i}__test.pt")
    for name, body in [("aa_mutator", 'stats["sequence_logprobs"][0][:] = 999\n        return np.zeros(len(stats["sequence_logprobs"]))'),
                       ("zz_reader", 'self.calls += 1\n        return np.array([-sum(x) for x in stats["sequence_logprobs"]]) + self.calls')]:
        directory = tmp_path / "out/logs/mod1" / name
        directory.mkdir(parents=True)
        (directory / "frozen_estimator.py").write_text('''class HallucinationDetector(Estimator):
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")
        assert context["meta"]["eval_meta"]["vocab_size"] == 600
        self.calls = 0
    def __call__(self, stats):
        ''' + body + "\n")
    monkeypatch.setattr("sys.argv", ["evaluate", "--mode", "evaluate", "--data-root", str(data_root),
                                    "--cache-dir", str(cache), "--output", str(output), "--bootstrap", "3"])
    main()
    first = np.load(output / "d0__test_scores.npz")
    last = np.load(output / "d8__test_scores.npz")
    np.testing.assert_array_equal(first["new/zz_reader"], first["original/SequenceLogProb"] + 1)
    np.testing.assert_array_equal(first["new/zz_reader"], last["new/zz_reader"])
    assert (output / "d8__test_bootstrap.npz").is_file()
    filtered = tmp_path / "new_method_only"
    monkeypatch.setattr("sys.argv", ["evaluate", "--mode", "evaluate", "--data-root", str(data_root),
                                    "--cache-dir", str(cache), "--output", str(filtered), "--bootstrap", "3",
                                    "--method-prefix", "new/zz_reader"])
    main()
    with np.load(filtered / "d0__test_scores.npz") as subset:
        assert set(subset.files) == {"labels", "duplicate_disjoint_mask", "claim_group_ids",
                                     "original/SequenceLogProb", "new/zz_reader"}
        np.testing.assert_array_equal(subset["new/zz_reader"], first["new/zz_reader"])
        np.testing.assert_array_equal(subset["original/SequenceLogProb"], first["original/SequenceLogProb"])


def test_transfer_selection_uses_source_validation_only(tmp_path):
    from scripts.evaluate_cached_claims import primary_paths
    paths = []
    cases = [("a1", "modelA", "evolution", "stratified", 75, 99),
             ("a2", "modelA", "evolution", "stratified", 80, 10),
             ("b1", "modelB", "evolution", "stratified", 79, 50),
             ("independent", "modelA", "independent", "stratified", 95, 99),
             ("grouped", "modelA", "evolution", "claim_group", 96, 99),
             ("pilot", "modelA", "evolution", "stratified", 97, 99)]
    for name, model, mode, split, validation, test in cases:
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "frozen_estimator.py"
        path.write_text("# frozen")
        (folder / "summary.json").write_text(json.dumps({"selected_validation_auc_percent": validation, "test_auc": test}))
        (folder / "trial_config.json").write_text(json.dumps({"search_mode": mode, "task": {"split_strategy": split}, "model": {"hf_model_id": model}}))
        paths.append(path)
    assert {p.parent.name for p in primary_paths(paths)} == {"a2", "b1"}


def test_header_ablation_preserves_causal_claim_likelihoods():
    from scripts.evaluate_cached_claims import trim_header_suffix
    from packing.evaluate.uncertainty_quantification.alignment_baselines import claim_likelihoods
    sample = {"input_ids": torch.tensor([100, 101, 8, 9, 10, 200, 201, 202, 203, 204]),
              "orig_input_ids": torch.tensor([8, 9, 10]), "sequence_logprobs": np.arange(9, dtype=float),
              "greedy_logprobs": np.arange(10), "top_k_logits": torch.zeros(10, 512)}
    trimmed = trim_header_suffix([sample], 4)[0]
    np.testing.assert_array_equal(claim_likelihoods(sample["sequence_logprobs"], sample["orig_input_ids"], sample["input_ids"]),
                                  claim_likelihoods(trimmed["sequence_logprobs"], trimmed["orig_input_ids"], trimmed["input_ids"]))
    assert len(trimmed["sequence_logprobs"]) == len(trimmed["input_ids"])-1
    assert len(sample["input_ids"]) == 10


def test_report_does_not_replace_validation_winner_after_test_failure():
    from scripts.report_campaign import select_primary_runs
    common = {"model": "writer", "search_mode": "evolution", "split": "stratified", "status": "finished"}
    rows = [{**common, "run": "strong_validation_failed_test", "validation_auc_percent": 80.},
            {**common, "run": "weaker_validation_valid_test", "validation_auc_percent": 79., "macro_auc_percent": 90.}]
    assert select_primary_runs(rows) == ["strong_validation_failed_test"]
    rows.append({**common, "run": "request_protocol_failure", "validation_auc_percent": 99., "protocol_excluded": True})
    assert select_primary_runs(rows) == ["strong_validation_failed_test"]


def test_held_out_chunk_timeout_restores_signal_handler():
    import signal
    import time
    from scripts.evaluate_cached_claims import score_with_timeout
    previous = signal.getsignal(signal.SIGALRM)
    def stalled(stats):
        time.sleep(1)
    with pytest.raises(TimeoutError):
        score_with_timeout(stalled, {}, .01)
    assert signal.getsignal(signal.SIGALRM) == previous
    np.testing.assert_array_equal(score_with_timeout(lambda stats: [1., 2.], {}, 1), [1., 2.])


def test_augmented_cohort_requires_unchanged_original_predictions(tmp_path):
    from scripts.verify_augmented_evaluation import verify_cohort
    original, extended = tmp_path/"original", tmp_path/"extended"
    original.mkdir(); extended.mkdir()
    new_method = "new/uncertainty_quantification_qwen35_4b_evo_1048"
    before, after = {}, {}
    for i in range(9):
        name = f"d{i}"
        before[name] = {"methods": {"baseline": {}}}
        after[name] = {"methods": {"baseline": {}, new_method: {}}}
        np.savez(original/(name+"_scores.npz"), labels=[0,1], baseline=[.1,.9])
        np.savez(extended/(name+"_scores.npz"), labels=[0,1], baseline=[.1,.9], **{new_method:[.2,.7]})
    (original/"all_datasets.json").write_text(json.dumps(before))
    (extended/"all_datasets.json").write_text(json.dumps(after))
    assert verify_cohort(original,extended)["common_method_predictions_identical"]
    np.savez(extended/"d8_scores.npz", labels=[0,1], baseline=[.1,.8], **{new_method:[.2,.7]})
    with pytest.raises(ValueError, match="Original predictions changed"):
        verify_cohort(original,extended)
    assert not (extended/"cohort_verified.json").exists()


def test_documented_features_need_no_full_vocabulary_entropy():
    from packing.evaluate.uncertainty_quantification.supervised_baselines import (
        compact_features, DOCUMENTED_FEATURE_INDICES, FittedDetector,
    )
    stats = {"sequence_logprobs": [np.array([-1., -2., -3.])],
             "top_k_logits": [np.array([[2.,1.],[1.,0.],[3.,1.],[2.,0.]])],
             "token_entropies": [np.array([.1,.4,.2,.3])]}
    full = compact_features(stats)
    del stats["token_entropies"]
    restricted = compact_features(stats, "documented")
    assert full.shape == (1,17) and restricted.shape == (1,14)
    np.testing.assert_array_equal(restricted, full[:,DOCUMENTED_FEATURE_INDICES])
    class Model:
        mod1_feature_set = "documented"
        def predict_proba(self, X):
            assert X.shape == (1,14)
            return np.array([[.3,.7]])
    np.testing.assert_array_equal(FittedDetector(Model())(stats), [.7])


def test_incremental_cohort_preserves_scores_and_rejects_changed_reference(tmp_path):
    from scripts.merge_evaluation_cohorts import merge_cohorts
    from scripts.verify_augmented_evaluation import verify_cohort
    original, additional, merged = [tmp_path/name for name in ["original", "additional", "merged"]]
    original.mkdir(); additional.mkdir()
    baseline, old_method, new_method = "original/SequenceLogProb", "new/old", "new/extra"
    old, extra = {}, {}
    for i in range(9):
        name = f"d{i}"
        old[name] = {"methods": {baseline: {"auc_percent": 75}, old_method: {"auc_percent": 80}}}
        extra[name] = {"methods": {baseline: {"auc_percent": 75}, new_method: {"auc_percent": 85}}}
        common = dict(labels=np.array([0,1,0,1]), duplicate_disjoint_mask=np.ones(4,dtype=bool),
                      claim_group_ids=np.arange(4), **{baseline: np.array([.1,.8,.5,.7])})
        np.savez(original/(name+"_scores.npz"), **common, **{old_method: np.array([.4,.7,.3,.9])})
        np.savez(additional/(name+"_scores.npz"), **common, **{new_method: np.array([.2,.9,.4,.8])})
        np.savez(original/(name+"_bootstrap.npz"), **{baseline: [.7,.8], old_method: [.8,.9]})
        np.savez(additional/(name+"_bootstrap.npz"), **{baseline: [.7,.8], new_method: [.9,.95]})
    (original/"all_datasets.json").write_text(json.dumps(old))
    (additional/"all_datasets.json").write_text(json.dumps(extra))
    assert merge_cohorts(original,additional,merged)["reference_recomputed_and_identical"]
    assert verify_cohort(original,merged,new_prefix=new_method)["common_method_predictions_identical"]
    with np.load(merged/"d8_bootstrap.npz") as saved:
        np.testing.assert_array_equal(saved[old_method], [.8,.9])
        np.testing.assert_array_equal(saved[new_method], [.9,.95])
    np.savez(additional/"d8_bootstrap.npz", **{baseline: [.6,.8], new_method: [.9,.95]})
    with pytest.raises(ValueError, match="bootstrap draws differ"):
        merge_cohorts(original,additional,merged)
    assert not (merged/"all_datasets.json").exists()
    assert not (merged/"cohort_verified.json").exists()


def test_fitted_auc_scores_preserve_rank_when_probabilities_saturate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from packing.evaluate.uncertainty_quantification.supervised_baselines import (
        FittedDetector, compact_features, DOCUMENTED_FEATURE_NAMES,
    )
    model = LogisticRegression()
    model.mod1_feature_set = "documented"
    model.classes_ = np.array([0,1])
    model.n_features_in_ = len(DOCUMENTED_FEATURE_NAMES)
    model.coef_ = np.zeros((1,model.n_features_in_))
    model.coef_[0,DOCUMENTED_FEATURE_NAMES.index("nll_sum")] = 1.
    model.intercept_ = np.zeros(1)
    stats = {"sequence_logprobs": [np.array([-40.]),np.array([-50.])],
             "top_k_logits": [np.ones((2,2)),np.ones((2,2))]}
    probabilities = model.predict_proba(compact_features(stats,"documented"))[:,1]
    assert roc_auc_score([0,1],probabilities) == .5
    scores = FittedDetector(model)(stats)
    np.testing.assert_array_equal(scores,[40.,50.])
    assert roc_auc_score([0,1],scores) == 1.
