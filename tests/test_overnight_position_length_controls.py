"""Focused fixtures for exact candidate parity, fixed contrasts and the nine gate."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import overnight_position_length_controls as audit


def test_frozen_source_and_float32_reference_contract():
    candidate, _ = audit.instantiate({})
    assert audit.sha(audit.SOURCE) == audit.SOURCE_SHA
    sequences = [np.array([], dtype=np.float16), np.array([-2.5], dtype=np.float16),
                 np.array([-1., -4., -2.], dtype=np.float16),
                 np.array([-.1, -10., -3., -.125], dtype=np.float16)]
    expected = candidate({'sequence_logprobs': sequences}).astype(np.float64)
    controls = audit.control_scores(sequences)
    mass = np.array([0., 1., 1.25, np.sum((np.arange(4, dtype=np.float32) / 3) ** 2)], dtype=np.float32)
    # Reconstructing a normalized score is mathematically consistent; the saved
    # unnormalized frozen reference itself, never this product, is parity-gated.
    assert expected[0] == 0 and controls['float32_quadratic_mean'][0] == 0
    assert expected[1] == controls['float32_quadratic_mean'][1] == 2.5
    assert np.allclose(expected[1:], controls['float32_quadratic_mean'][1:] * mass[1:])
    assert tuple(controls) == audit.CONTROLS
    assert all(values.dtype == np.float64 for values in controls.values())
    assert controls['float32_nll_sum'][2] == 7
    assert controls['float32_nll_mean'][2] == float(np.float32(7) / np.float32(3))


def test_constant_surprise_separates_length_and_weight_mass():
    sequences = [np.full(n, -2., dtype=np.float16) for n in [1, 2, 5, 17]]
    candidate, _ = audit.instantiate({})
    c = audit.control_scores(sequences)
    assert np.array_equal(c['lp_total_length'], [1., 2., 5., 17.])
    assert np.array_equal(c['float32_nll_sum'], [2., 4., 10., 34.])
    assert np.array_equal(c['float32_nll_mean'], [2.] * 4)
    assert np.array_equal(c['float32_quadratic_mean'], [2.] * 4)
    assert len(np.unique(candidate({'sequence_logprobs': sequences}))) > 1
    # Later position, not just length, changes quadratic mean under permutation.
    pair = audit.control_scores([np.array([-1., -9.]), np.array([-9., -1.])])
    assert np.array_equal(pair['float32_nll_mean'], [5., 5.])
    assert np.array_equal(pair['float32_quadratic_mean'], [9., 1.])


def test_nonfinite_inputs_fail_without_variant_selection():
    with pytest.raises(ValueError, match='finite one-dimensional'):
        audit.control_scores([np.array([-1., np.nan])])


def test_lp_gate_reproduces_full_pack_scores_and_no_topk_access(monkeypatch):
    from tests.test_overnight_batch_sensitivity import toy_raw
    from packing.evaluate.uncertainty_quantification.features import make_all_stats
    from scripts.evaluate_cached_claims import score_with_timeout
    raw = toy_raw(19)
    candidate, baseline = audit.instantiate(raw['meta'])
    stats, _, _, _ = make_all_stats(raw, input_regime='logits')
    expected = dict(frozen_quadratic_sum=score_with_timeout(candidate, stats, 30),
                    canonical_nll=score_with_timeout(baseline, stats, 30))
    # Proves the proposed run does not require/pack any top512 tensors.
    for sample in raw['samples']:
        del sample['top_k_logits'], sample['top_k_indices']
    actual = audit.score_raw(raw, 'reference')
    assert all(audit.exact(actual[key], expected[key]) for key in expected)
    controls = audit.score_raw(raw, 'controls')
    assert tuple(controls) == audit.METHODS
    assert all(audit.exact(controls[key], expected[key]) for key in expected)


def test_identity_guard_rejects_label_and_mask_changes():
    frame = pd.DataFrame({'claim': ['Alpha', 'Beta', ' alpha '], 'is_hallu': [0, 1, 0]})
    raw = {'samples': [dict(claim=c.strip(), label=y) for c, y in zip(frame.claim, frame.is_hallu)]}
    training = pd.DataFrame({'claim': ['ALPHA']})
    reference = dict(labels=np.array([0, 1, 0]), claim_group_ids=np.array([0, 1, 0]),
                     duplicate_disjoint_mask=np.array([False, True, False]))
    assert set(audit.validate_rows(raw, frame, training, reference, 3)) == set(reference)
    reference['duplicate_disjoint_mask'][0] = True
    with pytest.raises(ValueError, match='not byte-identical'):
        audit.validate_rows(raw, frame, training, reference, 3)
    reference['duplicate_disjoint_mask'][0] = False
    raw['samples'][1]['label'] = 0
    with pytest.raises(ValueError, match='ordering mismatch'):
        audit.validate_rows(raw, frame, training, reference, 3)


def test_all_nine_reference_gate_precedes_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'OUTPUT', tmp_path)
    (tmp_path / 'reference').mkdir()
    (tmp_path / 'PLAN.json').write_text('{}')
    plan = {'datasets': [dict(name=f'dataset{i}', raw_sha256=f'raw{i}', reference_npz_sha256=f'npz{i}') for i in range(9)]}
    for entry in plan['datasets']:
        scores = tmp_path / 'reference' / (entry['name'] + '_scores.npz')
        scores.write_bytes(b'fixture-only')
        receipt = dict(status='passed_reference', source_sha256=audit.SOURCE_SHA, raw_sha256=entry['raw_sha256'],
                       reference_npz_sha256=entry['reference_npz_sha256'], plan_sha256=audit.sha(tmp_path / 'PLAN.json'),
                       labels_groups_masks_exact=True, frozen_quadratic_sum_byte_exact=True, canonical_nll_byte_exact=True,
                       scores_sha256=audit.sha(scores))
        (tmp_path / 'reference' / (entry['name'] + '.json')).write_text(json.dumps(receipt))
    audit.require_reference(plan)
    last = tmp_path / 'reference/dataset8.json'
    receipt = json.loads(last.read_text()); receipt['frozen_quadratic_sum_byte_exact'] = False
    last.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='All-nine byte-parity gate failed'):
        audit.require_reference(plan)
    receipt['frozen_quadratic_sum_byte_exact'] = True; last.write_text(json.dumps(receipt))
    (tmp_path / 'reference/dataset8_scores.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='All-nine byte-parity gate failed'):
        audit.require_reference(plan)


def test_paired_claim_draws_preserve_duplicate_units_and_strict_scope(monkeypatch):
    # Existing weighted-bootstrap utility supplies one multiplicity per claim;
    # all methods and strict sensitivity share the very same draws.
    monkeypatch.setattr(audit, 'BOOTSTRAPS', 20)
    labels = np.array([0, 0, 1, 1])
    arrays = dict(labels=labels, claim_group_ids=np.array([0, 0, 1, 2]),
                  duplicate_disjoint_mask=np.array([False, False, True, True]), __dataset_name__='fixture')
    arrays.update({key: np.array([1., 1., 2., 3.]) for key in audit.METHODS})
    boot = audit.paired_bootstraps(arrays)
    assert len(boot) == 2 * len(audit.METHODS)
    for scope in ('all', 'strict'):
        ref = boot[scope + '/canonical_nll']
        assert all(np.array_equal(boot[scope + '/' + key], ref, equal_nan=True) for key in audit.METHODS)
    assert np.isnan(boot['strict/canonical_nll']).all()  # no invented AUC for one class
