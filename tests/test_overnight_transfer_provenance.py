import importlib.util
import json
from pathlib import Path
import copy

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overnight_transfer_eval", ROOT / "scripts/overnight_transfer_eval.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cached_transfer_dry_plan_records_actual_tokenizer_only_directory(tmp_path, monkeypatch, capsys):
    # Empty cache markers suffice for a dry plan: no model, tokenizer or cache
    # is opened for inference/scoring in this test.
    cache, directory = tmp_path / "target_cache", tmp_path / "report"
    cache.mkdir()
    directory.mkdir()
    registry = directory / "frozen_manifest.json"
    registry.write_text(json.dumps(dict(methods=[])))
    backup = ROOT / "data/cache_backup_manifest.json"
    entries = json.loads(backup.read_text())
    names = [Path(name).name.removesuffix(".gz") for name in entries
             if Path(name).parent.name == "mod1-transfer-cache"]
    assert len(names) == 6
    for name in names:
        (cache / name).touch()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "audit_target_caches", lambda *args: (_ for _ in ()).throw(AssertionError("cache audit called during dry plan")))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess started during dry plan")))
    monkeypatch.setattr("sys.argv", ["transfer", "--directory", str(directory), "--registry", str(registry),
        "--cache-dir", str(cache), "--data-root", str(ROOT / "data/claims"), "--cache-manifest", str(backup),
        "--fitted-models", str(ROOT / "out/research20261021/supervised_grouped/models.joblib"),
        "--kind", "supervised"])
    module.main()
    capsys.readouterr()
    command_path = next((directory / "transfer_llama32_1b").glob("*/commands.json"))
    plan = json.loads(command_path.read_text())
    assert plan["execute"] is False and len(plan["commands"]) == 6
    for row in plan["commands"]:
        command = row["command"]
        assert command[command.index("--mode") + 1] == "evaluate"
        assert command[command.index("--tokenizer-id") + 1] == "/workspace/mod1-models/llama31-tokenizer"
    protocol = json.loads((command_path.parent / "transfer_protocol.json").read_text())
    assert protocol["shared_tokenizer_directory"] == "/workspace/mod1-models/llama31-tokenizer"
    assert protocol["shared_tokenizer_model"] == module.TOKENIZER_ID
    assert protocol["shared_tokenizer_revision"] == module.TOKENIZER_REVISION


def test_partial_primary_is_not_opened_transferred_or_added_to_full_cohort(tmp_path, monkeypatch, capsys):
    cache, directory = tmp_path / "target_cache", tmp_path / "report"
    cache.mkdir()
    directory.mkdir()
    source = tmp_path / "inputs/full.py"
    source.parent.mkdir()
    source.write_text("# synthetic frozen source, not executed\n")
    full = dict(method="new/full", role="primary", full_budget=True, source_sha256=module.sha(source),
                staged_file="inputs/full.py", phase="evolution_screen", policy="neutral",
                writer_model="Qwen/Qwen3.5-9B", seed=2107, requests=60, planned_requests=60)
    registry = directory / "frozen_manifest.json"
    registry.write_text(json.dumps(dict(methods=[full])))
    backup = ROOT / "data/cache_backup_manifest.json"
    for name in json.loads(backup.read_text()):
        if Path(name).parent.name == "mod1-transfer-cache":
            (cache / Path(name).name.removesuffix(".gz")).touch()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess started during dry plan")))
    monkeypatch.setattr("sys.argv", ["transfer", "--directory", str(directory), "--registry", str(registry),
        "--cache-dir", str(cache), "--data-root", str(ROOT / "data/claims"), "--cache-manifest", str(backup),
        "--fitted-models", str(tmp_path / "missing.joblib"), "--kind", "new"])
    module.main()
    capsys.readouterr()
    first_path = next((directory / "transfer_llama32_1b").glob("*/commands.json"))
    first_commands = json.loads(first_path.read_text())["commands"]
    partial = {**full, "method": "new/partial", "full_budget": False, "requests": 20,
               "source_sha256": "0"*64, "staged_file": "MUST_NOT_OPEN_MISSING_PARTIAL.py"}
    registry.write_text(json.dumps(dict(methods=[full, partial])))
    module.main()
    capsys.readouterr()
    paths = list((directory / "transfer_llama32_1b").glob("*/commands.json"))
    assert paths == [first_path]
    assert json.loads(first_path.read_text())["commands"] == first_commands
    protocol = json.loads((first_path.parent / "transfer_protocol.json").read_text())
    assert [row["method"] for row in protocol["source_method_identities"]] == ["new/full"]
    assert [row["method"] for row in protocol["excluded_partial_primary_identities"]] == ["new/partial"]
    report = json.loads((first_path.parent / "transfer_report.json").read_text())
    assert report["excluded_partial_primary_identities"] == [partial]
    assert not any(row["method"] == "new/partial" for row in report["methods"])


def test_full_nine_requires_all_427_rows_exact_identity_finite_and_passed_numerics():
    receipt=dict(status="passed",rows=427,historical_cache_unchanged=True,
        quality_labels_used_for_numerical_guard=False,quality_auc_computed=False,
        comparison=dict(status="passed",all_rows_compared=427,
            all_claims_labels_ids_input_ids_attention_masks_shapes_dtypes_exact=True,
            all_compared_numeric_values_finite=True,labels_used_for_guard=False,quality_auc_computed=False,
            guard_checks=dict(observed_logprob_gross_drift=True,canonical_nll_rank_preservation=True),
            absolute_difference_summaries=dict(sequence_logprobs=dict(max_abs_difference=.0078125)),
            canonical_nll=dict(spearman_rho=.99999)))
    assert module.completion_numerical_reference_passed(dict(numerical_reference_preflight=receipt))
    assert not module.completion_numerical_reference_passed({})
    for key,value in [("status","failed"),("rows",426),("historical_cache_unchanged",False),("quality_auc_computed",True)]:
        bad=copy.deepcopy(receipt);bad[key]=value
        assert not module.completion_numerical_reference_passed(dict(numerical_reference_preflight=bad))
    for key,value in [("all_rows_compared",426),("all_claims_labels_ids_input_ids_attention_masks_shapes_dtypes_exact",False),("all_compared_numeric_values_finite",False)]:
        bad=copy.deepcopy(receipt);bad["comparison"][key]=value
        assert not module.completion_numerical_reference_passed(dict(numerical_reference_preflight=bad))
    for maximum,rho in [(.101,.99999),(.01,.9989),(.01,float("nan"))]:
        bad=copy.deepcopy(receipt)
        bad["comparison"]["absolute_difference_summaries"]["sequence_logprobs"]["max_abs_difference"]=maximum
        bad["comparison"]["canonical_nll"]["spearman_rho"]=rho
        assert not module.completion_numerical_reference_passed(dict(numerical_reference_preflight=bad))


def test_nine_cache_audit_rejects_otherwise_pinned_receipt_without_numeric_check_six_unchanged(tmp_path,monkeypatch):
    from types import SimpleNamespace
    completion=dict(status="complete_three_missing_caches",target=dict(model=module.TARGET_ID,revision=module.TARGET_REVISION,
        all_files_match_pinned_historical_hashes=True),tokenizer=dict(model=module.TOKENIZER_ID,revision=module.TOKENIZER_REVISION,
        all_files_match_pinned_historical_hashes=True),tokenizer_reference_parity=dict(exact_chat_input_ids_match=True,
        exact_raw_claim_input_ids_match=True),all_six_historical_caches_unchanged=True)
    path=tmp_path/"completion.json";path.write_text(json.dumps(completion))
    args=SimpleNamespace(completion_receipt=path,directory=tmp_path,coverage="nine")
    monkeypatch.setattr(module,"ROOT",tmp_path)
    with pytest.raises(ValueError,match="427-row mixed-runtime numerical reference"):
        module.audit_target_caches(args,[],[],{})
    args.coverage="six"
    result=module.audit_target_caches(args,[],[],{})
    assert result["completion_pin_verified"] is True and result["numerical_reference_preflight_verified"] is False
