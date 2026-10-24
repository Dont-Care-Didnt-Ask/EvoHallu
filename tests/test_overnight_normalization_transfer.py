"""CPU fixtures for immutable normalization transfer; no raw-cache inference."""
import copy
import datetime as dt
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from scripts import overnight_normalization_transfer as m


def rows():
    return dict(labels=np.array([0,1,0,1,0,1]), claim_group_ids=np.array([0,1,2,3,4,5]),
                duplicate_disjoint_mask=np.array([True,True,False,True,True,True]))


def values():
    return {**rows(),m.ORIGINAL:np.array([.1,.7,.2,.8,.3,.9]),m.FIXED:np.array([.2,.6,.3,.9,.1,.8]),m.NLL:np.array([.3,.7,.4,.6,.2,.8])}


@pytest.fixture
def candidates(monkeypatch):
    original=Path.glob
    monkeypatch.setattr(Path,'glob',lambda self,pattern:iter(()) if str(self)=='out/campaign' else original(self,pattern))
    return m.load_three({})


def stats():
    return dict(sequence_logprobs=[np.array([-1-i*.25,-.2,-2,.0,-.7,-3-i*.5,-.1],dtype=np.float16) for i in range(8)])


def test_lp_dependency_and_source_normalizer_are_exact():
    c=m.lp_dependency_contract()
    assert c['normalizer_calls']==4 and c['median_calls_in_normalizer']==2
    assert m.sha(m.frozen.parent.SOURCE)==m.SOURCE_SHA
    assert m.sha(Path(m.frozen.__file__))==m.HELPER_SHA
    assert m.sha(m.SOURCE/'normalization.json')==m.WEIGHTS_SHA


def test_original_and_proxy_compile_into_separate_private_namespaces(candidates):
    original,fixed,_=candidates
    a=original.__call__.__globals__;b=fixed.estimator.__call__.__globals__
    assert a is not b
    before=original(stats());fixed(stats())
    assert m.frozen.parent.exact(before,original(stats()))
    assert a['np'] is np and b['np'] is np


def test_saved_source_statistics_used_without_fit_and_zero_mads_preserved(candidates,monkeypatch):
    monkeypatch.setattr(m.frozen,'fit',lambda *a:pytest.fail('fit must never be called'))
    original,fixed,_=candidates
    whole=fixed(stats())
    single=np.concatenate([fixed(dict(sequence_logprobs=[x])) for x in stats()['sequence_logprobs']])
    assert m.frozen.parent.exact(whole,single)
    assert fixed.statistics[2]['mad']==fixed.statistics[3]['mad']==0
    assert fixed.statistics[0]['median']==141 and fixed.statistics[1]['mad']==2.734375
    assert not m.frozen.parent.exact(whole,original(stats()))


def test_real_worker_fixture_checks_source_parity_then_all_four_target_cases(tmp_path,candidates,monkeypatch):
    import torch
    import pandas as pd
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'OUTPUT',tmp_path/'study')
    monkeypatch.setattr(m,'assert_protected',lambda x:None)
    m.OUTPUT.mkdir();(m.OUTPUT/'PLAN.json').write_text('{}')
    samples=[]
    for i,lp in enumerate(stats()['sequence_logprobs']):
        ids=torch.arange(len(lp)+1)
        samples.append(dict(claim=f'fixture{i}',label=i%2,input_ids=ids,orig_input_ids=ids,sequence_logprobs=lp,
                            greedy_logprobs=np.r_[lp,lp[-1]],token_entropies=np.zeros(len(lp)+1,dtype=np.float16)))
    raw=dict(samples=samples,meta={})
    cache=tmp_path/'data/restored/mod1-transfer-cache/a.pt';cache.parent.mkdir(parents=True);torch.save(raw,cache)
    csv=tmp_path/'claims.csv';pd.DataFrame({'claim':[s['claim'] for s in samples],'is_hallu':[s['label'] for s in samples]}).to_csv(csv,index=False)
    train=tmp_path/'data/claims/popqa/popqa_train_filtered_10shot.csv';train.parent.mkdir(parents=True);pd.DataFrame({'claim':['unrelated training claim']}).to_csv(train,index=False)
    _,identities=m.rows_from_csv(csv,train)
    original,fixed,nll=candidates
    orig,base=m.frozen.parent.score_case(original,nll,raw,'original_512')
    fixed_scores,_=m.frozen.parent.score_case(fixed,nll,raw,'original_512')
    gold=tmp_path/'source.npz';np.savez(gold,**identities,original_primary512=orig,original_512=fixed_scores,canonical_nll=base)
    target_reference=tmp_path/'target_reference.npz';np.savez(target_reference,**identities,**{m.provenance.BASELINE:base})
    e=dict(name='a',rows=len(samples),index=0,csv='claims.csv',source_cache=str(cache.relative_to(tmp_path)),source_raw_sha256=m.sha(cache),source_scores='source.npz',historical_six=True,six_nll_reference='target_reference.npz')
    plan=dict(datasets=[e]);m.worker(plan,0,'source',m.OUTPUT/'source')
    m.validate_finished(m.OUTPUT/'source',e,'source',m.sha(cache))
    cohort=m.OUTPUT/'six';cohort.mkdir();(cohort/'cache_audit.json').write_text(json.dumps({'per_dataset':[{'dataset':'a','raw_sha256':m.sha(cache)}]}))
    m.worker(plan,0,'target',cohort/'target')
    result=json.loads((cohort/'target/a.json').read_text())
    assert len(result['frozen_cases'])==4
    assert len({x['score_sha256'] for x in result['frozen_cases']})==1
    m.validate_finished(cohort/'target',e,'target',m.sha(cache))
    np.savez(gold,**identities,original_primary512=orig+1,original_512=fixed_scores,canonical_nll=base)
    with pytest.raises(ValueError,match='LP-only source scores'):
        m.worker(plan,0,'source',m.OUTPUT/'bad_source')
    assert not (m.OUTPUT/'bad_source/a_scores.npz').exists()


def test_bootstrap_is_shared_deterministic_and_strict_groups_are_remapped():
    a=values();first=m.bootstrap_arrays(a,'fixture',count=12);second=m.bootstrap_arrays(a,'fixture',count=12)
    assert all(np.array_equal(first[k],second[k],equal_nan=True) for k in first)
    b=copy.deepcopy(a);b[m.FIXED]=b[m.NLL].copy()
    output=m.bootstrap_arrays(b,'fixture',count=12)
    assert all(np.array_equal(output[s+'/'+m.FIXED],output[s+'/'+m.NLL],equal_nan=True) for s in ['full','strict'])
    # Removed groups must not remain in the resampling population.
    b['claim_group_ids']=np.array([5,20,99,200,201,501])
    remapped=m.bootstrap_arrays(b,'fixture',count=12)
    assert all(np.array_equal(output[k],remapped[k],equal_nan=True) for k in output)


def test_partial_dataset_coverage_never_creates_macro():
    v=values();b=m.bootstrap_arrays(v,'a')
    result=m.metric_summary({'a':v},{'a':b},['a','b'])
    assert result['status']=='partial_no_macro' and 'scopes' not in result


def test_fold_each_dataset_before_macro_and_identical_methods_have_zero_paired_difference():
    v=values();v[m.FIXED]=v[m.NLL].copy();b=m.bootstrap_arrays(v,'a')
    result=m.metric_summary({'a':v},{'a':b},['a'])
    for scope in ['full','strict']:
        for style in ['directed','folded']:
            x=result['scopes'][scope][style]['fixed_minus_nll']
            assert x['delta_pp']==0 and x['paired_claim_interval']['ci95_pp']==[0,0]
    assert set(result['per_dataset'])=={'a'}


def test_row_dtypes_masks_and_group_identity_are_strict():
    a=rows();m.require_rows(a,rows())
    for key in m.ROW_KEYS:
        b=rows();b[key]=b[key][::-1]
        with pytest.raises(ValueError,match='Labels'):m.require_rows(b,a)
    b=rows();b['labels']=b['labels'].astype(np.int32)
    with pytest.raises(ValueError):m.require_rows(b,a)


def test_nine_audit_rejects_missing_formal427_guard_before_opening_cache(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'assert_protected',lambda x:None)
    receipt=tmp_path/'completion.json';receipt.write_text(json.dumps({'status':'complete_three_missing_caches'}))
    monkeypatch.setattr(m.provenance,'audit_target_caches',lambda *a:pytest.fail('raw caches must stay unopened'))
    with pytest.raises(ValueError,match='427-row'):
        m.target_audit(dict(datasets=[],completion_receipt='completion.json'),'nine',tmp_path/'nine')


def test_copy_once_is_byte_exact_and_never_overwrites(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'immutable six\n');dest=tmp_path/'nine/copy'
    fingerprint=m.sha(source);m.copy_once(source,dest,fingerprint);m.copy_once(source,dest,fingerprint)
    assert dest.read_bytes()==source.read_bytes()
    dest.write_bytes(b'conflict')
    with pytest.raises(ValueError,match='overwrite'):m.copy_once(source,dest,fingerprint)
    assert source.read_bytes()==b'immutable six\n'


def test_six_receipt_and_report_never_change_when_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'OUTPUT',tmp_path/'study')
    six=m.OUTPUT/'six/target';six.mkdir(parents=True)
    for suffix in ['_scores.npz','_bootstrap.npz','.json']:(six/('a'+suffix)).write_bytes(('frozen'+suffix).encode())
    (m.OUTPUT/'six/TRANSFER_REPORT.json').write_text('{}')
    manifest=m.freeze_six();m.reuse_six(dict(datasets=[dict(name='a',historical_six=True)]))
    assert m.assert_six_immutable()==manifest
    assert all((m.OUTPUT/'nine/target'/p.name).read_bytes()==p.read_bytes() for p in six.iterdir())
    (m.OUTPUT/'six/TRANSFER_REPORT.json').write_text('changed')
    with pytest.raises(ValueError,match='Immutable'):m.reuse_six(dict(datasets=[]))


def test_stale_prediction_checksum_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'OUTPUT',tmp_path);(tmp_path/'PLAN.json').write_text('{}')
    e=dict(name='a');folder=tmp_path/'target';folder.mkdir()
    for suffix in ['_scores.npz','_bootstrap.npz']:(folder/('a'+suffix)).write_bytes(b'fixture')
    receipt=dict(status='complete',dataset='a',phase='target',raw_sha256='raw',weights_sha256=m.WEIGHTS_SHA,
        source_sha256=m.SOURCE_SHA,plan_sha256=m.sha(tmp_path/'PLAN.json'),transparent_proxy_original_byte_exact=True,
        target_fit=False,bootstrap_samples=500,frozen_cases=[dict(case=c,nll_byte_exact=True) for c in m.CASES],
        scores_sha256=m.sha(folder/'a_scores.npz'),bootstrap_sha256=m.sha(folder/'a_bootstrap.npz'))
    (folder/'a.json').write_text(json.dumps(receipt));m.validate_finished(folder,e,'target','raw')
    (folder/'a_scores.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):m.validate_finished(folder,e,'target','raw')


def test_external_reference_missing_stays_pending(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'NIGHT',tmp_path/'night')
    result=m.external_parity(dict(datasets=[dict(name='a',historical_six=False)]),{'a':values()},{'a':m.bootstrap_arrays(values(),'a')},['a'],'nine')
    assert result['status']=='pending_main_target_reference'


def test_deadline_kills_only_dummy_owned_cpu_child(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        m.run_bounded([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'child.log',m.Bounds(m.now()+dt.timedelta(seconds=.1)))
    assert time.monotonic()-started<4


def test_cancel_and_fixed_hard_deadline(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'NIGHT',tmp_path);(tmp_path/'watchdog.cancel').touch()
    with pytest.raises(InterruptedError):m.Bounds(m.now()+dt.timedelta(minutes=1)).check()
    assert m.DEADLINE==dt.datetime(2026,10,22,2,3,tzinfo=dt.timezone.utc)
    assert m.MAX_SECONDS==1200 and m.COUNT==500


def test_protected_source_fit_state_drift_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    source=tmp_path/'normalization.json';source.write_text('{"median":141}')
    plan=dict(protected_sha256={'normalization.json':m.sha(source)})
    m.assert_protected(plan)
    source.write_text('{"median":0}')
    with pytest.raises(ValueError,match='Protected source artifact changed'):
        m.assert_protected(plan)
