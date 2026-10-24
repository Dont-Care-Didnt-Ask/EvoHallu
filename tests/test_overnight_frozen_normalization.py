"""CPU fixtures for unchanged-source median capture/freeze semantics."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from scripts import overnight_frozen_normalization as control

@pytest.fixture
def candidate_factory(monkeypatch):
    original=Path.glob
    def skip_heads(self,pattern):return iter(()) if str(self)=='out/campaign' else original(self,pattern)
    monkeypatch.setattr(Path,'glob',skip_heads)
    return lambda:control.load_candidate({})

def raw_fixture(n=19):
    rows=[]
    for i in range(n):
        size=12+i%7;lp=np.array([-0.1-((i+1)*(t+3)%23)/4 for t in range(size)],dtype=np.float16);ids=torch.arange(size+1)
        rows.append(dict(claim=f'row{i}',label=i%2,input_ids=ids,orig_input_ids=ids,sequence_logprobs=lp,
                         greedy_logprobs=np.r_[lp,lp[-1]],token_entropies=np.zeros(size+1,dtype=np.float16),
                         top_k_logits=torch.zeros((size+1,512),dtype=torch.float16),top_k_indices=torch.zeros((size+1,512),dtype=torch.long)))
    return dict(samples=rows,meta={})

def sequence_stats(raw):return dict(sequence_logprobs=[x['sequence_logprobs'] for x in raw['samples']])

def test_immutable_source_contract():
    d=control.source_contract();assert d['normalizer_calls']==4 and not d['source_edited'] and not d['ast_rewritten']

def test_transparent_proxy_preserves_original_predictions(candidate_factory):
    candidate,baseline=candidate_factory();raw=raw_fixture();original=control.parent.score_case(candidate,baseline,raw,'original_512',full_pack=True)
    wrapped=control.ProxiedEstimator(candidate);captured=control.parent.score_case(wrapped,baseline,raw,'original_512',full_pack=True)
    assert all(control.parent.exact(a,b) for a,b in zip(original,captured))
    assert len(wrapped.last_features)==4

def test_captured_features_are_pointwise_even_when_original_norm_is_not(candidate_factory):
    candidate,_=candidate_factory();wrapped=control.ProxiedEstimator(candidate);raw=raw_fixture()
    wrapped(sequence_stats(raw));whole=[x.copy() for x in wrapped.last_features];pieces=[[] for _ in range(4)]
    for sample in raw['samples']:
        wrapped(dict(sequence_logprobs=[sample['sequence_logprobs']]))
        for i,x in enumerate(wrapped.last_features):pieces[i].append(x)
    assert all(control.parent.exact(np.concatenate(parts),expected) for parts,expected in zip(pieces,whole))

def test_train_frozen_scores_are_nontrivial_and_exactly_chunk_order_invariant(candidate_factory):
    candidate,baseline=candidate_factory();raw=raw_fixture(529);capture=control.ProxiedEstimator(candidate);capture(sequence_stats(raw))
    fitted=control.fit_statistics(capture.last_features);frozen=control.ProxiedEstimator(candidate,fitted);arrays=[]
    for case in control.CASES:arrays.append(control.parent.score_case(frozen,baseline,raw,case))
    assert np.ptp(arrays[0][0])>0
    assert all(all(control.parent.exact(a,b) for a,b in zip(arrays[0],current)) for current in arrays)

def test_statistics_json_roundtrip_preserves_numeric_dtype_and_scores(candidate_factory):
    candidate,_=candidate_factory();stats=sequence_stats(raw_fixture());wrapped=control.ProxiedEstimator(candidate);wrapped(stats);fitted=control.fit_statistics(wrapped.last_features)
    first=control.ProxiedEstimator(candidate,fitted)(stats);second=control.ProxiedEstimator(candidate,json.loads(json.dumps(fitted)))(stats)
    assert control.parent.exact(first,second)
    assert all(x['dtype']=='<f4' for x in fitted)

def test_existing_degenerate_mad_rule_is_preserved(candidate_factory):
    candidate,_=candidate_factory();fitted=[dict(dtype='<f4',median=0.,mad=0.) for _ in range(4)]
    values=control.ProxiedEstimator(candidate,fitted)(sequence_stats(raw_fixture()))
    assert np.array_equal(values,np.zeros_like(values))

def test_proxy_is_private_and_restores_namespace_after_error(candidate_factory):
    candidate,_=candidate_factory();namespace=candidate.__call__.__globals__;original=namespace['np'];median=np.median
    with pytest.raises(IndexError):control.ProxiedEstimator(candidate,[])(sequence_stats(raw_fixture()))
    assert namespace['np'] is original and np.median is median

def test_unexpected_median_calls_fail_closed():
    proxy=control.MedianProxy()
    for _ in range(8):proxy.median(np.array([1.,2.]))
    with pytest.raises(ValueError,match='additional median'):proxy.median(np.array([1.,2.]))

def test_all9_proxy_references_gate_stale_cache_and_missing_data(tmp_path,monkeypatch):
    monkeypatch.setattr(control,'OUTPUT',tmp_path);(tmp_path/'reference').mkdir();plan=dict(datasets=[dict(name=f'd{i}',raw_sha256=f'raw{i}') for i in range(9)])
    for e in plan['datasets']:
        d=dict(status='passed_original_proxy_reference',parent_source_sha256=control.parent.SOURCE_SHA,raw_sha256=e['raw_sha256'],labels_groups_masks_exact=True,
               conditions=[dict(original512_byte_exact=True,canonical_nll_byte_exact=True)])
        (tmp_path/'reference'/(e['name']+'.json')).write_text(json.dumps(d))
    control.require_reference(plan)
    p=tmp_path/'reference/d8.json';d=json.loads(p.read_text());d['raw_sha256']='different';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='reference gate failed'):control.require_reference(plan)
