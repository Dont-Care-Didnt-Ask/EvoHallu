"""Small CPU fixtures: immutable program behavior and diagnostic safety gates."""
from pathlib import Path
import copy
import json
import numpy as np
import pytest
import torch

from scripts import overnight_batch_sensitivity as audit

@pytest.fixture
def estimators(monkeypatch):
    from scripts.evaluate_cached_claims import load_estimators
    original=Path.glob
    def without_campaign_heads(self,pattern):
        return iter(()) if str(self)=='out/campaign' else original(self,pattern)
    monkeypatch.setattr(Path,'glob',without_campaign_heads)
    values,_=load_estimators([audit.SOURCE],{})
    return values[audit.METHOD],values[audit.BASELINE]

def toy_raw(n=17):
    samples=[]
    for i in range(n):
        size=12+i%7
        lp=np.asarray([-0.1-((i+1)*(j+3)%19)/4 for j in range(size)],dtype=np.float16)
        ids=torch.arange(size+1)
        samples.append(dict(claim=f'claim {i}',label=i%2,input_ids=ids,orig_input_ids=ids,
                            sequence_logprobs=lp,greedy_logprobs=np.r_[lp,lp[-1]],token_entropies=np.zeros(size+1,dtype=np.float16),
                            top_k_logits=torch.zeros((size+1,512),dtype=torch.float16),top_k_indices=torch.zeros((size+1,512),dtype=torch.long)))
    return dict(samples=samples,meta={})

def test_source_is_exact_frozen_candidate():
    assert audit.sha(audit.SOURCE)==audit.SOURCE_SHA

def test_singleton_robust_normalization_is_zero(estimators):
    candidate,baseline=estimators
    score,nll=audit.score_case(candidate,baseline,toy_raw(1),'original_full')
    assert score.tolist()==[0.0]
    assert nll[0]>0

def test_dependency_gate_preserves_exact_program_scores(estimators):
    candidate,baseline=estimators;raw=toy_raw()
    gated=audit.score_case(candidate,baseline,raw,'original_512')
    packed=audit.score_case(candidate,baseline,raw,'original_512',full_pack=True)
    assert all(audit.exact(a,b) for a,b in zip(gated,packed))

def test_shuffle_restores_original_positions_and_nll(estimators):
    candidate,baseline=estimators;raw=toy_raw()
    scores,nll=audit.score_case(candidate,baseline,raw,'shuffle451_512')
    expected,base=audit.score_case(candidate,baseline,raw,'original_full')
    assert audit.exact(nll,base)
    # One full batch has the same median/MAD under permutation.
    assert audit.exact(scores,expected)
    assert np.array_equal(audit.permutation(17,'shuffle451_512'),np.random.default_rng(451).permutation(17))

def test_exact_rejects_dtype_and_tie_changes():
    a=np.array([1.,2.],dtype=np.float64)
    assert not audit.exact(a,a.astype(np.float32))
    b=a.copy();b[1]=np.nextafter(b[1],np.inf)
    assert not audit.exact(a,b)

def test_prevalence_keeps_original_order():
    labels=np.r_[np.zeros(512,dtype=int),np.ones(17,dtype=int)]
    p=audit.prevalence(labels)
    assert p['minimum']==0 and p['maximum']==1 and p['last_chunk_rows']==17

def test_all_nine_references_required_before_variants(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'OUTPUT',tmp_path);(tmp_path/'reference').mkdir()
    plan=dict(datasets=[dict(name=f'dataset{i}',raw_sha256=f'raw{i}') for i in range(9)])
    for e in plan['datasets']:
        receipt=dict(status='passed_reference',source_sha256=audit.SOURCE_SHA,raw_sha256=e['raw_sha256'],labels_groups_masks_exact=True,
                     conditions=[dict(case='original_512',exact_original512_scores=True,canonical_nll_byte_exact=True)])
        (tmp_path/'reference'/(e['name']+'.json')).write_text(json.dumps(receipt))
    audit.require_all_reference(plan)
    p=tmp_path/'reference/dataset8.json';receipt=json.loads(p.read_text());receipt['conditions'][0]['canonical_nll_byte_exact']=False;p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='reference gate failed'):audit.require_all_reference(plan)
