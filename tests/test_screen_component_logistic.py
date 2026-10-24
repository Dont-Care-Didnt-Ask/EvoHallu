import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import fit_screen_component_logistic as fit
from scripts import overnight_component_model as model
from scripts.prepare_screen_component_ablations import fixture


ROOT = Path(__file__).resolve().parents[1]


def prepared_pool(tmp_path):
    source_manifest = ROOT / fit.ABLATION_MANIFEST
    manifest = json.loads(source_manifest.read_text())
    for item in manifest["methods"]:
        dest = tmp_path / item["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item["file"], dest)
    copied_manifest = tmp_path / fit.ABLATION_MANIFEST
    copied_manifest.write_text(json.dumps(manifest))
    output = tmp_path / "refit"
    path, pool_manifest = fit.prepare_pool(copied_manifest, output, tmp_path)
    return output, path, pool_manifest


def test_all_eleven_full_minus_drop_features_short_branches_and_source_guards(tmp_path):
    output, pool_path, manifest = prepared_pool(tmp_path)
    pool = model.ComponentPool(tmp_path, "refit/component_pool.json", model.sha(pool_path))
    stats = fixture([0, 1, 3, 8, 17, 31, 48], np.float16)
    before = copy.deepcopy(stats)
    X = pool.transform(stats)
    assert X.shape == (7, 12) and len(set(pool.feature_names)) == 12
    assert pool.feature_names[-1] == "float64_total_nll"
    np.testing.assert_array_equal(X[:,-1],[-np.sum(np.asarray(lp,dtype=np.float64)) for lp in stats["sequence_logprobs"]])
    assert {name.split("/")[0] for name in pool.feature_names[:-1]} == {"original", "neutral", "diverse", "compositional"}
    for index, component in enumerate(manifest["components"]):
        full = pool.detectors[component["full"]](copy.deepcopy(stats))
        drop = pool.detectors[component["drop"]](copy.deepcopy(stats))
        np.testing.assert_array_equal(X[:,index], np.asarray(full,dtype=float)-np.asarray(drop,dtype=float))
    # Diverse fallbacks remain NLL in both programs, hence their difference is
    # exactly zero for short sequences; the separate NLL column retains signal.
    diverse = [i for i,name in enumerate(pool.feature_names) if name.startswith("diverse/")]
    np.testing.assert_array_equal(X[:3,diverse], 0)
    assert X[1,-1] != 0
    for key in stats:
        for old, current in zip(before[key], stats[key]):
            np.testing.assert_array_equal(old, current)
    # Prepared sources are immutable/idempotent, and a changed byte is rejected.
    original = output / "components/screen_original__full/frozen_estimator.py"
    original.write_text(original.read_text()+"\n# tampered\n")
    with pytest.raises(ValueError,match="path/hash"):
        model.ComponentPool(tmp_path, "refit/component_pool.json", model.sha(pool_path))


def test_fit_only_grouped_cv_serialization_real_estimator_loading_and_guards(tmp_path,monkeypatch):
    output, pool_path, manifest = prepared_pool(tmp_path)
    pool = model.ComponentPool(tmp_path,"refit/component_pool.json",model.sha(pool_path))
    stats = fixture([8+i%17 for i in range(90)])
    X = pool.transform(stats)
    # Distinct normalized claims ensure outer and all inner partitions disjoint.
    groups = np.array([f"claim {i}" for i in range(90)])
    y = np.array([i%2 for i in range(90)])
    train, val = np.arange(60), np.arange(60,90)
    fitted, predictions, report = fit.fit_head(X,y,groups,train,val)
    changed, changed_y = X.copy(), y.copy()
    changed[val] += 10000
    changed_y[val] = 1 - changed_y[val]
    again, _, again_report = fit.fit_head(changed,changed_y,groups,train,val)
    assert report["C_grid"] == [.01,.1,1.,10.,100.]
    assert report["chosen_C"] == again_report["chosen_C"]
    assert report["scaler_fit_samples"] == 60
    assert report["outer_validation_is_independent"] is False
    assert all(fold["shared_normalized_claims"] == 0 for fold in report["inner_folds"])
    np.testing.assert_array_equal(fitted[0].mean_,again[0].mean_)
    np.testing.assert_array_equal(fitted[-1].coef_,again[-1].coef_)
    artifact = dict(feature_names=pool.feature_names,pool_file="refit/component_pool.json",pool_sha256=model.sha(pool_path),
                    inference_module_sha256=model.sha(model.__file__),scaler_mean=fitted[0].mean_.tolist(),
                    scaler_scale=fitted[0].scale_.tolist(),coefficients_standardized=fitted[-1].coef_[0].tolist(),
                    intercept=float(fitted[-1].intercept_[0]))
    weights = output / "weights.json"
    weights.write_text(json.dumps(artifact))
    wrapper = output / "frozen/screen_component_logistic/frozen_estimator.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(fit.wrapper_source("refit/weights.json",model.sha(weights),model.sha(model.__file__)))
    monkeypatch.setattr(model,"ROOT",tmp_path)
    # Use the evaluator's real candidate parser to execute the serialized wrapper.
    detector = model.load_detector(wrapper,{"supervised":False,"meta":{"eval_meta":{}}})
    np.testing.assert_allclose(detector(stats),predictions,atol=1e-10,rtol=1e-10)
    from scripts.evaluate_cached_claims import load_estimators
    monkeypatch.chdir(tmp_path)
    estimators, provenance = load_estimators([wrapper], eval_meta={})
    np.testing.assert_allclose(estimators["new/screen_component_logistic"](stats), predictions, atol=1e-10, rtol=1e-10)
    assert provenance["new/screen_component_logistic"]["sha256"] == model.sha(wrapper)
    np.testing.assert_allclose(np.concatenate([detector({key:values[:45] for key,values in stats.items()}),
                                             detector({key:values[45:] for key,values in stats.items()})]),predictions,atol=1e-10,rtol=1e-10)
    with pytest.raises(ValueError,match="module hash"):
        model.FrozenComponentLogistic({},"refit/weights.json",model.sha(weights),"0"*64)
    weights.write_text(weights.read_text()+" ")
    with pytest.raises(ValueError,match="path/hash"):
        model.load_detector(wrapper,{"supervised":False,"meta":{"eval_meta":{}}})
