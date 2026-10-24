import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import softmax

from packing.funsearch import programs_database as database


spec = importlib.util.spec_from_file_location(
    "overnight_sampler_trial", Path(__file__).resolve().parents[1] / "scripts/overnight_sampler_trial.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_linear_rank_formula_preserves_uniform_reserve_and_fitness_order():
    # Scores are intentionally not sorted, and have very different gaps.
    scores = np.array([-500.0, 100.0, -499.0, 0.0])
    got = module.rank_uniform_probabilities(scores)
    expected = .8 * np.array([1, 4, 2, 3]) / 10 + .2 / 4
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
    assert np.isfinite(got).all() and (got >= .2 / len(scores)).all()
    np.testing.assert_allclose(got.sum(), 1)
    np.testing.assert_allclose(module.rank_uniform_probabilities([8, 8, 8]), np.ones(3) / 3)
    np.testing.assert_array_equal(module.rank_uniform_probabilities([8]), [1.])
    # Linear ranks deliberately ignore fitness scale and translation.
    np.testing.assert_allclose(module.rank_uniform_probabilities(1e5 * scores + 3), got)


@pytest.mark.parametrize("scores", [[], [[1, 2]], [np.nan], [np.inf, 2]])
def test_invalid_fitness_rejected(scores):
    with pytest.raises(ValueError, match="nonempty finite"):
        module.rank_uniform_probabilities(scores)


def test_inherited_quantile_gate_short_source_selection_and_seed_replay_unchanged(monkeypatch):
    island = database.Island(functions_per_prompt=2, temp_sampling_flag=True, temp=40.)
    for score in range(10):
        island.register_program("x" * (score + 2), float(score), {})
    # Multiple programs in one score group exercise the separate length softmax.
    island.register_program("a much longer implementation" * 10, 9., {})
    original_softmax = database._softmax
    ordinary_calls = []

    def spy(logits, temperature):
        ordinary_calls.append((np.array(logits), temperature))
        return original_softmax(logits, temperature)

    monkeypatch.setattr(database, "_softmax", spy)
    np.random.seed(8421)
    inherited = island.get_prompt(.6)
    inherited_eligible_scores = ordinary_calls[0][0]
    # Gate is active: 40th percentile nearest is 4 and strictly greater survive.
    np.testing.assert_array_equal(inherited_eligible_scores, [5., 6., 7., 8., 9.])
    ordinary_calls.clear()
    with module.rank_parent_sampling(database) as calls:
        np.random.seed(8421)
        first = island.get_prompt(.6)
        np.random.seed(8421)
        replay = island.get_prompt(.6)
        assert first == replay
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[0]["eligible_scores"], inherited_eligible_scores)
        np.testing.assert_allclose(calls[0]["probabilities"], [.8 * r / 15 + .2 / 5 for r in range(1, 6)])
        # Only within-group softmax calls reach the inherited function.
        assert len(ordinary_calls) == 4
        assert all(temperature == 1.0 for _, temperature in ordinary_calls)
        for values, temperature in ordinary_calls:
            np.testing.assert_allclose(original_softmax(values, temperature), softmax(values))
    assert database._softmax is spy
    # Parent output format and increasing-fitness ordering are inherited.
    assert len(first) == len(inherited) == 5
    assert first[1] == sorted(first[1]) and set(first[1]).issubset({5., 6., 7., 8., 9.})


def test_small_population_skips_gate_and_database_island_choice_is_unchanged():
    from omegaconf import OmegaConf
    db = database.ProgramsDatabase(OmegaConf.create(dict(
        num_islands=2, functions_per_prompt=2, temp_sampling_flag=True, temp=40.)))
    for score in range(3):
        db.register_program("source" + str(score), float(score), None, {})
    np.random.seed(3001)
    original_island = db.get_prompt(percentile=.2)[2]
    with module.rank_parent_sampling(database) as calls:
        np.random.seed(3001)
        rank_result = db.get_prompt(percentile=.2)
    assert rank_result[2] == original_island
    np.testing.assert_array_equal(calls[0]["eligible_scores"], [0., 1., 2.])


def test_rank_wrapper_completes_synthetic_http_trial_and_records_failed_round(tmp_path):
    """Exercise the real wrapper/task with a stub response, never a real writer."""
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="next_experiments/next_evo_neutral_rank_uniform_2107")
    assert OmegaConf.to_container(cfg, resolve=True)["logs_dir"] == (
        "out/logs/research20261021/uncertainty_quantification_next_evo_neutral_rank_uniform_2107")
    samples = [dict(label=i % 2, claim=f"synthetic rank sample {i}",
                    input_ids=torch.arange(6), orig_input_ids=torch.arange(4),
                    sequence_logprobs=np.full(5, -1. - i % 2),
                    greedy_logprobs=np.full(6, -.5), token_entropies=np.ones(6),
                    top_k_logits=torch.zeros(6, 512),
                    top_k_indices=torch.zeros(6, 512, dtype=torch.long)) for i in range(40)]
    raw = tmp_path / "train.pt"
    torch.save(dict(samples=samples, meta={}), raw)
    cfg.task.train_raw_path = str(raw)
    cfg.task.test_raw_path = str(tmp_path / "MUST_NOT_OPEN.pt")
    cfg.logs_dir = str(tmp_path / "run")
    cfg.num_rounds = 3
    cfg.trial.max_seconds = 120
    received = []
    source = '''```python
class HallucinationDetector(Estimator):
    request_seed = SEED
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")
    def __call__(self, stats):
        return np.asarray([-np.mean(x) for x in stats["sequence_logprobs"]])
```'''

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            # Entire second round fails. The common failure watchdog permits
            # round three, and the parent-weight event still records round two.
            content = ("synthetic invalid detector" if 2207 <= payload["seed"] < 2307
                       else source.replace("SEED", str(payload["seed"])))
            body = json.dumps(dict(id="test", object="chat.completion", created=0, model="test",
                choices=[dict(index=0, finish_reason="stop", message=dict(role="assistant", content=content))],
                usage=dict(prompt_tokens=60, completion_tokens=40, total_tokens=100))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cfg.cluster.api_base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        module.main.__wrapped__(cfg)
    finally:
        server.shutdown()
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["status"] == "finished" and summary["stop_reason"] == "round_budget"
    assert summary["requests"] == len(received) == 6
    assert summary["valid_candidates"] == 4
    assert summary["held_out_test_evaluated"] is False
    events = [json.loads(line) for line in (tmp_path / "run/parent_sampling_events.jsonl").read_text().splitlines()]
    assert len(events) == 3
    assert all(event["uniform_reserve"] == .2 for event in events)
    provenance = json.loads((tmp_path / "run/parent_sampling_provenance.json").read_text())
    assert provenance["parent_sampling"] == "rank_uniform"
    assert provenance["test_labels_used"] is False
    assert (tmp_path / "run/frozen_estimator.py").is_file()
    assert database._softmax.__name__ == "_softmax"
