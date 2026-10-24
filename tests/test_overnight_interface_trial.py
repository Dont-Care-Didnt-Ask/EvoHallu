import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import prompt_policies


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overnight_interface_trial", ROOT / "scripts/overnight_interface_trial.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_contract_only_appends_truthful_interface_to_neutral_prompt_and_restores_hook():
    from packing.evaluate.uncertainty_quantification.task_uncertainty_quantification import system_prompt_fabric

    chat = [dict(role="system", content=system_prompt_fabric()), dict(role="user", content="Unchanged parent example")]
    untouched = json.loads(json.dumps(chat))
    original = prompt_policies.candidate_prompt
    neutral = original(chat, "neutral", 3, 1)
    with module.clarified_candidate_prompts(prompt_policies) as calls:
        actual = prompt_policies.candidate_prompt(chat, "neutral", 3, 1)
        assert actual[1:] == neutral[1:]
        assert actual[0]["content"] == neutral[0]["content"] + "\n\n" + module.CONTRACT
        assert "Build off a strong baseline:" not in actual[0]["content"]
        assert 'shape (L_i + 1, 512)' in module.CONTRACT
        assert 'stats["top_k_logits"][i][:-1]' in module.CONTRACT
        assert 'is_hallu=1' in module.CONTRACT and 'dictionary' in module.CONTRACT
        assert calls[0]["actual_prompt_sha256"] == module.chat_sha(actual)
        with pytest.raises(ValueError, match="requires the neutral"):
            prompt_policies.candidate_prompt(chat, "original")
    assert prompt_policies.candidate_prompt is original and chat == untouched
    with pytest.raises(RuntimeError):
        with module.clarified_candidate_prompts(prompt_policies):
            raise RuntimeError("test restoration on exception")
    assert prompt_policies.candidate_prompt is original


def test_actual_saved_prompts_and_hashes_include_contract_without_extra_requests(tmp_path):
    """Real task/spawned evaluator with ragged synthetic arrays and HTTP stub."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    import torch
    from hydra import compose, initialize_config_dir

    protected = [ROOT / name for name in ["src/experiments/extension_trial.py",
        "src/experiments/prompt_policies.py", "src/packing/funsearch/programs_database.py"]]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    original = prompt_policies.candidate_prompt
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="next_experiments/next_evo_neutral_interface_2107")
    samples = []
    for i in range(40):
        length = 6 + i % 3
        samples.append(dict(label=i % 2, claim=f"synthetic interface sample {i}",
            input_ids=torch.arange(length), orig_input_ids=torch.arange(length-2),
            sequence_logprobs=np.full(length-1, -1.-i%2, dtype=np.float16),
            greedy_logprobs=np.full(length, -.5, dtype=np.float16),
            token_entropies=np.ones(length, dtype=np.float16),
            top_k_logits=torch.zeros(length,512,dtype=torch.float16),
            top_k_indices=torch.zeros(length,512,dtype=torch.long)))
    raw = tmp_path / "train.pt"
    torch.save(dict(samples=samples, meta={}), raw)
    cfg.task.train_raw_path = str(raw)
    cfg.task.test_raw_path = str(tmp_path / "MUST_NOT_OPEN.pt")
    cfg.logs_dir = str(tmp_path / "run")
    cfg.num_rounds = 2
    cfg.trial.max_seconds = 150
    received = []
    source = '''```python
class HallucinationDetector(Estimator):
    request_seed = SEED
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")
    def __call__(self, stats):
        return np.asarray([-np.mean(np.asarray(x,dtype=np.float64)) for x in stats["sequence_logprobs"]])
```'''

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            assert payload["messages"][0]["content"].endswith(module.CONTRACT)
            body = json.dumps(dict(id="test",object="chat.completion",created=0,model="test",
                choices=[dict(index=0,finish_reason="stop",message=dict(role="assistant",
                    content=source.replace("SEED",str(payload["seed"]))))],
                usage=dict(prompt_tokens=60,completion_tokens=40,total_tokens=100))).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self,*args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    cfg.cluster.api_base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        module.main.__wrapped__(cfg)
    finally:
        server.shutdown()
    run = tmp_path / "run"
    summary = json.loads((run / "summary.json").read_text())
    assert summary["requests"] == len(received) == 4
    assert summary["valid_candidates"] == 4 and summary["total_tokens"] == 400
    assert summary["held_out_test_evaluated"] is False
    verification = json.loads((run / "interface_prompt_verification.json").read_text())
    assert verification["actual_saved_prompts_verified"] == 4
    assert verification["prompt_hashes_match_ledgers"] and verification["runtime_hook_restored"]
    trace = [json.loads(line) for line in (run / "interface_prompt_injections.jsonl").read_text().splitlines()]
    assert len(trace) == 4
    assert {row["actual_prompt_sha256"] for row in trace} == {module.chat_sha(p["messages"]) for p in received}
    assert prompt_policies.candidate_prompt is original
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in protected)
    assert (run / "frozen_estimator.py").exists()
