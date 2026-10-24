"""Independently check saved per-formula weights, fits and Table 3 numbers."""
from pathlib import Path
import datetime as dt
import hashlib
import json

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/table3_20261022"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def rank_auc(y, score):
    positive = y == 1
    m, n = int(positive.sum()), int((~positive).sum())
    assert m and n and np.isfinite(score).all()
    ranks = rankdata(score,method="average")
    return float(100*(ranks[positive].sum()-m*(m+1)/2)/(m*n))


def main():
    result = json.loads((OUT/"RESULTS.json").read_text())
    plan = json.loads((OUT/"PLAN.json").read_text())
    archive = json.loads((OUT/"ARTIFACT_SHA256.json").read_text())
    frozen = json.loads((OUT/"MODELS_FROZEN_BEFORE_TEST.json").read_text())
    for name, record in archive.items():
        assert digest(OUT/name)==record["sha256"],name
    for record in plan["inputs"]:
        assert digest(ROOT/record["file"])==record["sha256"],record["file"]
    lookup={(r["policy"],r["scope"],r["dataset"]):r for r in result["metrics"]}
    cells=0
    def check(policy,scope,name,y,evo,head):
        nonlocal cells
        row=lookup[(policy,scope,name)]
        a,b=rank_auc(y,evo),rank_auc(y,head)
        for key,value in [("evo_directed",a),("logreg_directed",b),("directed_delta",b-a),
                          ("evo_folded",max(a,100-a)),("logreg_folded",max(b,100-b)),
                          ("folded_delta",max(b,100-b)-max(a,100-a))]:
            assert abs(row[key]-value)<1e-10,(policy,name,key,row[key],value)
            cells+=1
    with np.load(OUT/"training_predictions_and_features.npz",allow_pickle=False) as z:
        y=z["labels"];fit=z["train_indices"];validation=z["validation_indices"];groups=z["normalized_claim_groups"]
        assert not set(groups[fit]) & set(groups[validation])
        for policy in plan["policies"]:
            model=result["models"][policy]
            assert digest(OUT/(policy+"_weights.json"))==frozen["weights"][policy]
            X=z[policy+"/features"]
            assert X.shape==(7007,len(plan["original_effective_coefficients"][policy]))
            np.testing.assert_allclose(model["scaler_mean"],X[fit].mean(axis=0),atol=1e-12)
            np.testing.assert_allclose(model["scaler_scale"],X[fit].std(axis=0),atol=1e-12)
            w=np.asarray(model["coefficients_raw"])
            np.testing.assert_allclose(w,np.asarray(model["coefficients_standardized"])/np.asarray(model["scaler_scale"]),atol=1e-12)
            prediction=X@w+model["intercept_raw"]
            np.testing.assert_allclose(prediction,z[policy+"/logreg"],rtol=1e-10,atol=1e-10)
            for scope,idx,name in [("fit_train",fit,"PopQA fit"),("diagnostic_validation",validation,"PopQA diagnostic validation")]:
                check(policy,scope,name,y[idx],z[policy+"/evo"][idx],z[policy+"/logreg"][idx])
            # Fresh CV reproduction from fitting rows only; rank-based AUROC
            # is independent of the producer's sklearn metric call.
            folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=441).split(X[fit],y[fit],groups[fit]))
            reproduced=[]
            for record in model["tuning"]:
                scores=[]
                for train,test in folds:
                    assert not set(groups[fit][train]) & set(groups[fit][test])
                    pipeline=make_pipeline(StandardScaler(),LogisticRegression(C=record["C"],max_iter=2000))
                    pipeline.fit(X[fit][train],y[fit][train])
                    scores.append(rank_auc(y[fit][test],pipeline.decision_function(X[fit][test])))
                np.testing.assert_allclose(scores,record["fold_auc"],rtol=0,atol=1e-10)
                reproduced.append((float(np.mean(scores)),record["C"]))
            assert max(reproduced,key=lambda pair:pair[0])[1]==model["chosen_C"]
            pipeline=make_pipeline(StandardScaler(),LogisticRegression(C=model["chosen_C"],max_iter=2000)).fit(X[fit],y[fit])
            np.testing.assert_allclose(pipeline[-1].coef_[0],model["coefficients_standardized"],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(pipeline[-1].intercept_[0],model["intercept_standardized"],atol=1e-10,rtol=1e-10)
    dataset_names=list(result["source_rows"])
    files=sorted((OUT/"source_nine").glob("*_scores.npz"))
    assert len(files)==9
    total=0
    for path in files:
        source=ROOT/"results/overnight20261021/screen_component_ablations/evaluations/source_nine"/path.name
        with np.load(path,allow_pickle=False) as z, np.load(source,allow_pickle=False) as old:
            for key in ("labels","duplicate_disjoint_mask","claim_group_ids"):
                np.testing.assert_array_equal(z[key],old[key])
            matches=[name for name in dataset_names if result["source_rows"][name]==len(z["labels"])]
            assert len(matches)==1
            name=matches[0];y=z["labels"];mask=z["duplicate_disjoint_mask"];total+=len(y)
            for policy in plan["policies"]:
                model=result["models"][policy]
                np.testing.assert_array_equal(z[policy+"/evo"],old["new/screen_"+policy+"__full"])
                raw_score=z[policy+"/features"]@np.asarray(model["coefficients_raw"])+model["intercept_raw"]
                np.testing.assert_allclose(raw_score,z[policy+"/logreg"],rtol=1e-10,atol=1e-10)
                check(policy,"source_nine",name,y,z[policy+"/evo"],z[policy+"/logreg"])
                check(policy,"strict_source_nine",name,y[mask],z[policy+"/evo"][mask],z[policy+"/logreg"][mask])
    assert total==97488
    for policy in plan["policies"]:
        for scope,macro in [("source_nine","macro_nine"),("strict_source_nine","strict_macro_nine")]:
            rows=[r for r in result["metrics"] if r["policy"]==policy and r["scope"]==scope]
            assert len(rows)==9
            for key in ("evo_directed","logreg_directed","directed_delta","evo_folded","logreg_folded","folded_delta"):
                assert abs(np.mean([r[key] for r in rows])-lookup[(policy,macro,"MACRO")][key])<1e-10
                cells+=1
    report=dict(status="passed",verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                validation_script_sha256=digest(Path(__file__)),result_sha256=digest(OUT/"RESULTS.json"),
                independent_rank_auc_metric_cells=cells,independently_reproduced_cv_fits=60,
                independently_reproduced_final_fits=4,frozen_evo_vectors_exact=36,
                source_dataset_rows=total,coefficient_unit_conversion_verified=True,
                original_input_hashes_unchanged=True,all_checks_passed=True)
    (OUT/"INDEPENDENT_VALIDATION.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
