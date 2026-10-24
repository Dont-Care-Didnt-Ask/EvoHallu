import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.extension_trial import build_prompt
from experiments.prompt_policies import candidate_prompt


def saved_chat():
    path = Path(__file__).resolve().parents[1] / "out/logs/mod1/uncertainty_quantification_qwen35_9b_evo_1048/prompt_0000.json"
    return json.loads(path.read_text())


def test_neutral_policy_preserves_parents_contract_and_original_chat():
    chat = saved_chat()
    original = json.loads(json.dumps(chat))
    assert candidate_prompt(chat) == original
    neutral = candidate_prompt(chat, "neutral")
    assert chat == original
    assert neutral[1] == original[1]
    assert "EXPECTED STRUCTURE" not in neutral[0]["content"]
    assert "Build off a strong baseline:" not in neutral[0]["content"]
    for text in ["shape (n_i-1,)", "shape (n_i, k)", "Output shape [N]", "Class named `HallucinationDetector`"]:
        assert text in neutral[0]["content"]


def test_diverse_policy_assigns_different_sibling_roles_and_is_repeatable():
    chat = saved_chat()
    a = candidate_prompt(chat, "diverse", 4, 0)
    b = candidate_prompt(chat, "diverse", 4, 1)
    assert a != b
    assert "Local-refinement role:" in a[0]["content"]
    assert "Exploration role:" in b[0]["content"]
    assert a[1] == b[1] == chat[1]
    assert candidate_prompt(chat, "diverse", 4, 1) == b


def test_frozen_context_never_samples_population_and_checks_round_count(tmp_path):
    class Forbidden:
        def get_prompt(self, **kwargs):
            raise AssertionError("sampled evolved population during a frozen-context experiment")
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps({"contexts": [{"chat": saved_chat()}]}))
    cfg = OmegaConf.create(dict(search_mode="frozen_context", seed=2107, num_rounds=1,
                                trial=dict(frozen_contexts_path=str(path))))
    chat, island = build_prompt(cfg, Forbidden(), "unused", 999, 0)
    assert chat == saved_chat() and island is None
    cfg.num_rounds = 2
    with pytest.raises(ValueError, match="rounds must equal"):
        build_prompt(cfg, Forbidden(), "unused", 999, 0)


def test_neutral_policy_rejects_unrecognized_prompt():
    with pytest.raises(ValueError, match="exactly one"):
        candidate_prompt([{"role": "system", "content": "Different format"}], "neutral")


def test_compositional_policy_keeps_contract_and_permits_many_features():
    chat = saved_chat()
    composite = candidate_prompt(chat, "compositional")
    assert composite[1] == chat[1]
    assert "EXPECTED STRUCTURE" not in composite[0]["content"]
    assert "no fixed feature-count limit" in composite[0]["content"]
    assert "Output shape [N]" in composite[0]["content"]
    assert candidate_prompt(chat, "compositional", 0, 0) == candidate_prompt(chat, "compositional", 0, 1)
