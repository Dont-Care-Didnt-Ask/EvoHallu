# Exploratory interface-and-metric clarification arm

Prepared after the training failure audit created **21 October 2026, 19:05:13 UTC**. This is a prospective additional arm, not a completed experiment. No validation or OOD score chooses the contract text. The main orchestrator decides whether sufficient authorized runtime remains; no automatic scheduling or GPU work is added here.

## Motivation and frozen evidence

The frozen audit contains **126 persisted candidates, 75 pipeline-valid and 51 invalid**. Its categories contain **36 ragged/shape/alignment/nonfinite failures**, **6 import/API failures**, and **5 inheritance/constructor failures**. These are different categories; “36/51 shape/API failures” is an imprecise shorthand. The source audit explicitly records that it did not inspect test data/predictions.

Evidence is preserved in `failure_audit_snapshot.json` and distilled into `training_failure_evidence.json`. Both are hashed in the separate study manifest. Examples include ragged NumPy conversion, length-28 versus length-27 broadcasting, `scipy.special.entropy`, missing Estimator inheritance, wrong constructor calls, and treating the input dictionary as `scipy.stats`. The design does not claim that every nonfinite result has one specific cause or that the proposed clarification will necessarily fix it.

## Sole intervention

Start from the existing neutral prompt and append the text in `interface_contract.txt` to its system message. The contract states the actual constructor and `__call__` API, batch dictionary and ragged array shapes, the extra top-k row, exact row alignment, raw-logit semantics, float16 numeric precautions, concrete SciPy API locations, finite output shape, and the existing directed score convention: larger score means `is_hallu=1` is more likely.

Because metric direction is included, the intervention is explicitly named **`interface_and_metric_contract`**. It is a bundled clarification, not a shape-only causal ablation. The direction statement comes from the provided Estimator/baseline contract; it is not chosen in response to validation or test performance.

For sample i, observed-token logprobs have length L_i and top-k arrays have shape `(L_i+1,512)`. Observed logprob entry t is calculated from logits row t, so `stats["top_k_logits"][i][:-1]` aligns with the whole observed-logprob vector. Batch sizes and token lengths differ; `stats` is a dictionary of lists of arrays, not a list of per-claim dictionaries. The constructor takes `context`, plus keyword arguments `stats_dependencies` and `level`. Top-512-normalized probabilities remain conditional approximations rather than known full-vocabulary probabilities.

All legal deterministic NumPy/SciPy formulas remain available. There is no new feature-count limit, prescribed baseline, required signal family, automatic repair, additional model call, fitted weight step, or change to inputs. Runtime evaluation, baseline fallback, candidate selection, islands, parent score groups, quantile filtering, softmax temperature, and within-score-group source-length selection remain inherited.

## Why the runtime hook is in prompt_policies

`extension_trial.run_trial` imports `candidate_prompt` into a local variable from `experiments.prompt_policies` inside the function. Setting an attribute named `extension_trial.candidate_prompt` would not alter that local import. The root orchestrator approved a narrowly scoped runtime replacement of `prompt_policies.candidate_prompt` instead. It first calls the original neutral transformation, appends only the contract, then restores the original function in `finally`.

No pinned core or prompt-policy file is edited. Actual request chats are saved by the unchanged runner. The wrapper verifies that every persisted request ends with exactly one complete contract and that each ledger prompt hash matches its actual saved chat. Separate injection events and contract/provenance receipts make the intervention reviewable.

## Matched design

| Item | Setting |
| --- | --- |
| Writer | Pinned Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Feature model and labels | Existing Llama3.1-8b-instruct cache; unchanged `is_hallu` |
| Prompt | Neutral plus interface-and-metric contract |
| Parent sampler | Inherited, unchanged |
| Seeds | 2107, 4319, 8741 |
| Budget | 30 rounds × 2 requests per seed; **180 requests total** |
| Generation settings | Same temperature 0.7, top-p 0.8, top-k 20, 4,096 output tokens |
| Training/validation split | Same historical stratified split and seed 441 |
| Winner | Same directed PopQA validation maximum with NLL fallback |
| Primary comparator | Ordinary neutral evolution, matched by writer/seed/request budget |
| Status | Prepared only; main must decide whether to execute |

The manifest is `results/next_experiments/interface_manifest.json`, alongside the existing readiness record. Its execution phase is `evolution_screen` for controller compatibility. It explicitly declares `analysis_phase=interface_control`; reporting must keep this arm separate from ordinary neutral evolution. **No analysis manifest, active execution manifest, existing configuration, or orchestration script was changed.** The main orchestrator must map this phase explicitly if it chooses to launch the new arm.

Primary diagnostics are execution validity and nonconstant training predictions per requested candidate. Report actual failure categories and all request/token costs. Secondary outcomes are the full-budget three-seed validation comparison, behavioural diversity on the fixed training probe, and frozen-method OOD results. A validity gain does not by itself prove an OOD-quality gain. Failure audits motivated the arm after the campaign began, so interpretation remains exploratory.

The longer input prompt changes prompt-token cost. Equal request counts do not imply equal token counts or elapsed time. Retain unsuccessful requests and do not refill failed candidates beyond the shared 60-request budget.

## Dry plan

```bash
.venv/bin/python scripts/run_overnight_parallel.py \
  --manifest results/next_experiments/interface_manifest.json \
  --phase all --ports 18000 18001
```

This is a dry run. The preparation task does not add `--execute`, start a model, or change the authorized global deadline. Extra orchestration remains the main agent's decision.

## Artifacts

- Wrapper: `scripts/overnight_interface_trial.py`.
- Three new configurations: `configs/next_experiments/next_evo_neutral_interface_{2107,4319,8741}.yaml`.
- Separate immutable manifest with core/model/data/config/wrapper/evidence hashes.
- Per-run `interface_contract_provenance.json`, `interface_contract.txt`, `interface_prompt_injections.jsonl`, and final `interface_prompt_verification.json` in addition to ordinary trial artifacts.
- Tests cover neutral-prompt preservation, hook restoration, exact saved-prompt injection, core-byte preservation, ragged synthetic inputs, request counts, and the unchanged real evaluator path. Synthetic HTTP responses do not count as writer-model experiments.

## Preparation validation

Two wrapper tests passed in **47.03 seconds**, including an actual two-round/four-request synthetic HTTP run through the real spawned evaluator: four valid candidates, four persisted chats containing exactly one contract, matching prompt hashes, restored runtime hook, and unchanged pinned core bytes. No held-out test file or real writer inference was used. All three Hydra configurations resolve to distinct expected log directories and 60 requests; every pinned source/config hash matches. The unchanged parallel controller accepts the separate 180-request manifest in dry-run mode.

Four CPU reporting regressions passed in **3.99 seconds**. The explicit `interface_metric_contract_vs_neutral_evolution` comparison pairs all three seeds only with the same writer and 60-request ordinary neutral evolution runs; different writer, request-budget, or partial runs cannot enter that contrast. The broader original-evolution reference remains separately named. This requires the orchestrator to map the new execution identities to `interface_control` before staging. Full machine-readable preparation evidence is in `validation.json`.
