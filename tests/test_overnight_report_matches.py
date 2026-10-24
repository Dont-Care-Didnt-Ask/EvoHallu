import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overnight_report", ROOT / "scripts/overnight_report.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def result(name, phase, policy, score, seed=1, writer="writerA", requests=60, full=True):
    return dict(method=name, status="complete", directed_macro_auc_percent=score-2,
        paper_macro_auc_percent=score, identity=dict(role="primary", phase=phase, policy=policy,
            writer_model=writer, seed=seed, requests=requests, full_budget=full,
            validation_auc_percent=score+5))


def test_neutral_controls_use_matching_writer_seed_budget_and_separate_original_reference():
    rows = [result("original", "evolution_screen", "original", 70),
            result("neutral", "evolution_screen", "neutral", 72),
            result("fixed", "feedback_control", "neutral", 71),
            result("rank", "sampler_control", "neutral", 73),
            result("neutralB", "evolution_screen", "neutral", 80, writer="writerB"),
            result("rankB", "sampler_control", "neutral", 79, writer="writerB"),
            result("neutral2", "evolution_screen", "neutral", 74, seed=2),
            result("rank2", "sampler_control", "neutral", 76, seed=2),
            result("neutral3", "evolution_screen", "neutral", 74, seed=3),
            result("wrong_budget", "sampler_control", "neutral", 90, seed=3, requests=30),
            result("neutral_partial", "evolution_screen", "neutral", 73, seed=4, full=False),
            result("rank_partial", "sampler_control", "neutral", 95, seed=4, full=False),
            result("screen_original", "prompt_screen", "original", 69, requests=30),
            result("screen_neutral", "prompt_screen", "neutral", 73, requests=30)]
    comparisons = module.matched_comparisons(rows)
    fixed = next(c for c in comparisons if c["comparison"] == "fixed_examples_vs_neutral_evolution")
    assert fixed["per_seed"][0]["reference"] == "neutral"
    assert fixed["paper_gain_pp"]["mean"] == -1
    rank_a = next(c for c in comparisons if c["comparison"] == "rank_sampling_vs_neutral_evolution" and c["writer_model"] == "writerA")
    assert [row["reference"] for row in rank_a["per_seed"]] == ["neutral", "neutral2"]
    assert rank_a["paper_gain_pp"]["mean"] == 1.5
    rank_b = next(c for c in comparisons if c["comparison"] == "rank_sampling_vs_neutral_evolution" and c["writer_model"] == "writerB")
    assert rank_b["per_seed"][0]["reference"] == "neutralB"
    assert rank_b["paper_gain_pp"]["mean"] == -1
    original_rank = next(c for c in comparisons if c["comparison"] == "original_evolution_reference" and c["phase"] == "sampler_control")
    assert original_rank["per_seed"][0]["reference"] == "original"
    assert original_rank["paper_gain_pp"]["mean"] == 3
    screen = next(c for c in comparisons if c["comparison"] == "fixed_context_prompt_reference")
    assert screen["per_seed"][0]["reference"] == "screen_original"
    assert screen["per_seed"][0]["requests"] == 30
    all_pairs = [p for c in comparisons for p in c["per_seed"]]
    assert all(p["requests"] == p["reference_requests"] and p["both_full_planned_budgets"] for p in all_pairs)
    assert not any(p["method"] in {"wrong_budget", "rank_partial"} for p in all_pairs)


def test_ambiguous_matching_reference_is_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        module.matched_comparisons([
            result("neutral1", "evolution_screen", "neutral", 70),
            result("neutral2", "evolution_screen", "neutral", 71)])


def test_interface_contract_matches_all_three_neutral_seeds_at_same_writer_and_budget():
    seeds = [2107, 4319, 8741]
    rows = []
    for i, seed in enumerate(seeds):
        rows.extend([result(f"original{seed}", "evolution_screen", "original", 69+i, seed=seed),
                     result(f"neutral{seed}", "evolution_screen", "neutral", 70+i, seed=seed),
                     result(f"interface{seed}", "interface_control", "neutral", 71+i, seed=seed)])
    rows.extend([result("other_writer", "interface_control", "neutral", 99, seed=2107, writer="writerB"),
                 result("wrong_budget", "interface_control", "neutral", 99, seed=4319, requests=30),
                 result("partial", "interface_control", "neutral", 99, seed=8741, requests=20, full=False)])
    comparisons = module.matched_comparisons(rows)
    interface = next(c for c in comparisons if c["comparison"] == "interface_metric_contract_vs_neutral_evolution")
    assert interface["reference_phase"] == "evolution_screen" and interface["reference_policy"] == "neutral"
    assert [pair["seed"] for pair in interface["per_seed"]] == seeds
    assert [pair["reference"] for pair in interface["per_seed"]] == [f"neutral{s}" for s in seeds]
    assert all(pair["requests"] == pair["reference_requests"] == 60 for pair in interface["per_seed"])
    assert interface["paper_gain_pp"]["mean"] == 1
    original = next(c for c in comparisons if c["comparison"] == "original_evolution_reference" and c["phase"] == "interface_control")
    assert [pair["reference"] for pair in original["per_seed"]] == [f"original{s}" for s in seeds]
    assert original["paper_gain_pp"]["mean"] == 2


def test_primary_seed_groups_exclude_partial_unknown_and_secondary_records():
    full1 = result("full1", "evolution_screen", "neutral", 70, seed=1)
    full2 = result("full2", "evolution_screen", "neutral", 72, seed=2)
    partial = result("partial", "evolution_screen", "neutral", 99, seed=3, requests=20, full=False)
    unknown = result("unknown", "evolution_screen", "neutral", 98, seed=4)
    unknown["identity"].pop("full_budget")
    secondary = result("trainbest", "evolution_screen", "neutral", 97, seed=5)
    secondary["identity"]["role"] = "trainbest"
    pending = result("full_pending_test", "evolution_screen", "neutral", 96, seed=6)
    pending["status"] = "pending_or_failed"
    groups = module.all_seed_primary_groups([full1, full2, partial, unknown, secondary, pending])
    assert len(groups) == 1
    assert groups[0]["staged_seeds"] == [1, 2, 6]
    assert groups[0]["evaluated_seeds"] == [1, 2]
    assert groups[0]["paper_macro"]["n"] == 2
    assert groups[0]["paper_macro"]["mean"] == 71
