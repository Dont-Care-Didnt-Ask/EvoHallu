"""Exact parent-weight and two-draw accounting on simple fitness pools."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("parent_analysis", Path(__file__).resolve().parents[1] / "scripts/overnight_parent_sampling_analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def event(scores, probabilities):
    return dict(eligible_scores=scores, probabilities=probabilities, inherited_temperature=40., uniform_reserve=.2)


def test_two_group_probabilities_do_not_change_selected_group_set():
    row = analysis.analyse_event(event([60, 80], [11/30, 19/30]))
    assert not row["one_group_degenerate"] and not row["group_set_selection_can_change"]
    assert row["rank_top_group_probability"] == pytest.approx(19/30)
    assert row["rank_two_parent_top_group_inclusion_probability"] == pytest.approx(1)
    assert row["inherited_two_parent_top_group_inclusion_probability"] == pytest.approx(1)
    assert row["two_parent_expected_mean_fitness_delta"] == pytest.approx(0)
    assert row["rank_two_parent_expected_mean_fitness"] == pytest.approx(70)
    assert row["two_parent_set_total_variation_distance"] == pytest.approx(0)
    assert row["rank_two_parent_set_normalized_entropy"] is None


def test_rank_weighting_can_strengthen_or_weaken_exploitation():
    stronger = analysis.analyse_event(event([70, 71, 72], [1/5, 1/3, 7/15]))
    weaker = analysis.analyse_event(event([0, 50, 100], [1/5, 1/3, 7/15]))
    assert stronger["expected_fitness_delta"] > 0 and stronger["entropy_delta"] < 0
    assert weaker["expected_fitness_delta"] < 0 and weaker["entropy_delta"] > 0
    assert 0 < stronger["total_variation_distance"] < 1
    probabilities = np.array([1/5, 1/3, 7/15])
    inclusion = analysis.inclusion_two(probabilities)
    assert inclusion.sum() == pytest.approx(2)
    assert np.all((inclusion > 0) & (inclusion < 1))
    pairs, probabilities = analysis.two_parent_sets(np.array([1/3, 1/3, 1/3]))
    assert pairs == [(0,1), (0,2), (1,2)]
    assert probabilities == pytest.approx([1/3, 1/3, 1/3])
    assert stronger["rank_two_parent_set_normalized_entropy"] < stronger["inherited_two_parent_set_normalized_entropy"]


def test_degenerate_events_excluded_and_invalid_recording_rejected():
    row = analysis.analyse_event(event([74.3], [1]))
    assert row["one_group_degenerate"] and row["rank_normalized_entropy"] is None
    result = analysis.summarize([row])
    assert result["excluded_one_group_events"] == 1
    assert result["nondegenerate_event_means"]["total_variation_distance"] is None
    with pytest.raises(ValueError, match="differ from the pinned"):
        analysis.analyse_event(event([70, 80], [.5, .5]))
