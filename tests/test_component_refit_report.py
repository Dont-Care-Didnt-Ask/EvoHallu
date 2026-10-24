import copy

import numpy as np
import pytest

from scripts import overnight_component_refit_report as report


def saved():
    labels=np.array([0,1,0,1,0,1,0,1])
    return dict(labels=labels,claim_group_ids=np.arange(8)//2,duplicate_disjoint_mask=np.ones(8,dtype=bool),
                **{report.METHOD:np.array([0,.8,.2,.9,.3,.7,.4,.6]),
                   report.BASELINE:np.array([0,.8,.2,.9,.3,.7,.65,.6])})


def test_complete_prespecified_scopes_and_paired_transfer_intervals():
    names=[f"dataset{i}" for i in range(9)]
    scores={name:saved() for name in names}
    boots={name:{report.METHOD:np.linspace(.71,.89,500),report.BASELINE:np.linspace(.61,.79,500)} for name in names}
    partial=report.scope_summary({name:scores[name] for name in names[:8]},boots,names)
    assert partial["verified_datasets"]==8 and "macro" not in partial
    source=report.scope_summary(scores,boots,names)
    assert source["status"]=="complete_predictions"
    assert source["paired_claim_bootstrap"]["gain_directed"]["ci95_pp"]==pytest.approx([10,10])
    boots[names[0]][report.METHOD][0]=np.nan
    counts=report.bootstrap_scope(boots,names)
    assert counts["gain_directed"]["finite_replicates"]==499
    target={name:copy.deepcopy(scores[name]) for name in names[:6]}
    same_boot={name:copy.deepcopy(boots[name]) for name in names[:6]}
    transfer=report.transfer_summary(scores,target,boots,same_boot,names[:6])
    assert transfer["matched_datasets"]==6
    assert all(value==0 for value in transfer["macro"].values())
    assert transfer["paired_claim_bootstrap"]["head_directed"]["ci95_pp"]==[0,0]
    assert transfer["paired_claim_bootstrap"]["gain_change_paper"]["ci95_pp"]==[0,0]
    assert transfer["per_dataset"][0]["paired_claim_bootstrap"]["head_directed"]["ci95_pp"]==[0,0]
    target[names[0]]["claim_group_ids"]=target[names[0]]["claim_group_ids"][::-1]
    with pytest.raises(ValueError,match="row identity"):
        report.transfer_summary(scores,target,boots,same_boot,names[:6])


def test_saved_scores_require_exact_rows_reference_baseline_and_protocol(tmp_path):
    folder,reference=tmp_path/"new",tmp_path/"reference"
    folder.mkdir();reference.mkdir()
    scores=saved();name="synthetic"
    np.savez_compressed(folder/(name+"_scores.npz"),**scores)
    np.savez_compressed(reference/(name+"_scores.npz"),**scores)
    identities={name:{key:scores[key] for key in report.ROW_KEYS}}
    errors=[]
    loaded,_,audit,_=report.load_scope(folder,[name],identities,reference,{"status":"passed"},errors)
    assert list(loaded)==[name] and audit[0]["status"]=="verified" and not errors
    loaded,_,audit,_=report.load_scope(folder,[name],identities,reference,{"status":"pending"},errors)
    assert loaded=={} and audit[0]["status"]=="pending_reference_or_protocol"
    changed=copy.deepcopy(scores);changed[report.BASELINE][0]+=.1
    np.savez_compressed(reference/(name+"_scores.npz"),**changed)
    loaded,_,audit,_=report.load_scope(folder,[name],identities,reference,{"status":"passed"},errors)
    assert loaded=={} and errors and audit[0]["status"]=="MISMATCH_or_invalid"
    (folder/(name+"_scores.npz")).write_bytes(b"incomplete npz upload")
    loaded,_,audit,_=report.load_scope(folder,[name],identities,reference,{"status":"passed"},[])
    assert loaded=={} and audit[0]["status"]=="possibly_copying"
    csv_path=tmp_path/"claims.csv"
    csv_path.write_text('claim,is_hallu\n" A   claim ",0\n"a CLAIM",1\n"other",0\n')
    actual,receipt=report.row_identities(csv_path,{"a claim"})
    np.testing.assert_array_equal(actual["claim_group_ids"],[0,0,1])
    np.testing.assert_array_equal(actual["duplicate_disjoint_mask"],[False,False,True])
    assert receipt["dataset_csv_sha256"]==report.sha(csv_path)
