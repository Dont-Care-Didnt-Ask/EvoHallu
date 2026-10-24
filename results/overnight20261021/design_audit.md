# Overnight experiment design audit

Prepared 21 October 2026 from the current repository and saved historical results. This is a design/code audit, not a new inference result. No GPU requests were issued and no shared experiment configuration was changed by this audit.

## Main decision

The prepared 1,020-request Qwen3.5-9B suite answers the user's immediate questions more directly than a longer single search. Complete matched policies and seed blocks first; spend remaining GPU time on an explicitly matched second-writer confirmation or missing feature-model caches. Evaluate frozen methods on the nine cached datasets while generation continues.

The feature model remains Llama3.1-8b-instruct. Qwen writes detector code. Neither new claim generation nor replacing the feature model is implicit in the main study.

## Priority order under an eight-hour limit

| Priority | Experiment | Concrete output |
| --- | --- | --- |
| 1 | Four-policy, 15-context, two-sibling frozen-context screen (120 requests) | Executability, truncation, sibling correlation, new nonconstant rankings per request, proposal training/validation quality |
| 2 | Original / neutral / diverse / compositional evolution, each with three seeds and 60 requests (720 requests) | Policy-by-seed winner validation AUROC, validity, useful diversity, and equal-budget search curves |
| 3 | Neutral fixed-NLL-example control, three seeds and 60 requests (180 requests) | Does evolutionary feedback help under the same neutral prompt and request budget? |
| 4 | Nine-dataset evaluation of every frozen run winner, grouped supervised-14 heads, and fixed reference methods | Both directed and paper-compatible macro AUROC; paired deltas and duplicate-disjoint diagnostics |
| 5 | Matched second-writer confirmation, original versus one policy selected by source validation | Whether a promising policy effect transfers to another writer |
| 6 | Frozen-formula transfer on the six existing Llama-3.2-1B caches; complete missing caches only if cheap | Source/target on identical dataset subsets, with frozen source-selected coefficients |

Generation phases can be scheduled differently from the prepared manifest to finish full seed blocks early. Preserve the same configurations, budgets and seeds, and record actual ordering. With only one GPU, finish priorities 1–4 before starting a second writer. With two GPUs, run independent clients/model replicas rather than tensor-parallelizing the 9B writer.

Historical timings are approximately 20 minutes per 60-request 9B run and 78 minutes per 60-request 27B run. These are old end-to-end observations, not guaranteed runtime. One evaluation worker in the prepared study may change the balance. Two-writer confirmation with two policies, two seeds and 60 requests each would be 240 requests, roughly 5.2 single-GPU-hours or 2.6 wall hours on two equally fast independent GPUs, before setup overhead. Start it only after measuring the actual host.

## What each contrast actually tests

- A (`original`) versus B (`neutral`) changes mandatory named-baseline-plus-one-signal guidance while preserving examples and evaluator interface.
- B versus C (`diverse`) adds local-refinement/exploration roles. This is a role-plus-family intervention; it does not independently isolate the effect of role assignment from the rotating feature-family hints.
- B versus D (`compositional`) adds multi-feature guidance without a feature-count ceiling. This tests prompted composition, not a separate mechanism for fitting coefficients.
- Neutral evolution versus neutral fixed-NLL examples changes adaptive examples while holding the formula prompt and request budget fixed. It does not isolate every difference between population sampling and independent sampling.
- The fixed-context screen isolates proposals conditional on historical examples. It does not establish improved end-to-end evolution. Its contexts all come from one historical 9B trajectory, so generalization to other parent distributions needs the evolution phase.

No complexity penalty should be added. The user's Claude illustration motivates testing complex complementary combinations, and source length alone is not a scientific quality criterion.

## Selection and reporting contract

1. Training directed AUROC controls population fitness and parent sampling.
2. PopQA validation directed AUROC selects the run winner, with the existing baseline fallback and first-occurrence tie rule.
3. Freeze winner identities and source SHA-256 before nine-dataset scoring.
4. Report all three seeds per policy. Mean and spread across seeds are primary; a validation-selected best seed may be an additional clearly labelled result.
5. Keep ordinary directed AUROC and compute the paper-compatible score per method and dataset as `max(AUC, 100-AUC)`, then average. Do not change `is_hallu` or fold the macro average.
6. The author's reporting clarification resolves baseline reproduction. It does not prove folded fitness was used during original search; do not silently alter search fitness.
7. The existing nine datasets have been examined before. New results are meaningful exploratory OOD comparisons, not a fresh blind benchmark. Search-seed variation and fixed-program bootstrap uncertainty are separate quantities.

The paper-compatible folded score permits a dataset-specific direction chosen using labels. It is useful for numerical comparison with the paper but is not evidence that a deployed detector can know the correct score direction on an unseen dataset. Keep directed results beside it.

## High-value CPU additions

**Evaluate the new grouped supervised-14 heads on all nine caches.** PopQA validation 82.17 for logistic regression versus 73.71 for NLL on that same grouped split is not a nine-dataset result. This evaluation needs no writer inference and supplies a strong quality reference for the new policy study. Include the existing stratified heads as distinct controls rather than overwriting them.

**Compare selection rules after freezing the proposal set.** Export the validation-selected winner and train-selected winner from every run and, if budget permits, a top-three validation collection. Freeze all identities before evaluation. This can answer whether the candidate pool contains generalizing methods that single-winner selection misses. It does not authorize choosing a method by its test performance.

**Inspect implementations, not only narratives.** Historical Qwen27B seed1048 disables its added gap feature through a length-equality guard. Check new high-ranked programs for dead terms, constants, accidental transforms of NLL, top-k normalization, dtype overflow, and early returns. A new source hash is not proof of a new ranking.

**Run short, exact ablations of actual winning terms.** Select the few source-validation winners and define ablations from their executable expressions before test scoring. Keep fixed coefficients. Distinguish deleting one component from refitting the remaining components.

**Use duplicate-disjoint diagnostics.** Historical train/validation shares 88 normalized claims and PopQA train/test shares 412 unique claims. Preserve original split for the matched main comparison, report disjoint test masks, and treat new claim-grouped searches as a separate arm if undertaken.

## Analysis details that matter

- Compare novelty conditional on validity and novelty per requested candidate. A policy can appear diverse by producing fewer executable programs; validity is part of its cost.
- Report the number of usable sibling pairs. Sibling correlation based only on pairs with two valid nonconstant outputs is a selected subset.
- The prepared 512-row training probe is diagnostic, not a held-out evaluation set. Its rank fingerprints can merge functions that differ outside the probe.
- Near-previous frequency grows with archive size. Compare at matched request prefixes (10, 20, 40, 60) and retain absolute distinct-ranking counts per request.
- Spearman >=0.995 detects positive near-duplicates. An inverse ranking is distinct for directed deployment but equivalent under the paper's direction-invariant measure. Report both signed and absolute correlation where interpreting folded quality.
- Distinct ranking is not necessarily useful: accompany diversity with training improvement, validation quality and OOD scores.
- Report actual output-token counts, truncation fraction, failure families, valid programs per GPU-hour, and end-to-end wall time. Extra sketches or repairs must count as requests and tokens.
- A 60-request arm with 30 valid programs is not equal to another arm with 60 valid programs. Match issued-request budgets, not successful-output counts.
- Three seeds are a small comparison. Show all seeds rather than relying on formal significance claims from three observations.
- No universal AUROC gain or paper-record improvement is guaranteed. The meaningful result may be a null policy effect, useful diversity with unchanged quality, or strong fitted controls outperforming generated formulas.

## Implementation pitfalls found

### Multiple shortlist programs collide in the current evaluator

`scripts/evaluate_cached_claims.py:load_estimators` names each frozen method as `new/` plus `path.parent.name`. Standard `.../RUN/frozen_estimator.py` is unique. Multiple files under `validation_shortlist` all receive `new/validation_shortlist` and overwrite one another, including files from different runs. Before evaluating top-three collections, stage each program in its own uniquely named directory or implement an explicit unique method identifier. Verify the final method count against the manifest.

### `--primary-only` collapses policy arms

`primary_paths` groups evolution runs by the writer's `hf_model_id`. It has no policy key. With all policy runs using Qwen3.5-9B, this keeps just the highest-validation run across all policies and seeds. Omit `--primary-only` for the prospective policy comparison and explicitly pass its frozen-source glob. Report all runs.

### The transfer wrapper is historical and must not be launched unchanged

`scripts/transfer_feature_model.py` uses old `/dev/shm` and `/workspace` paths, rotates writer checkpoint directories, and defaults to the historical primary cohort. Adapt paths and frozen-method selection explicitly before use. Source and target tokenizers have identical token-to-ID vocabularies but their merges differ; the historical protocol deliberately used the source tokenizer to hold actual input tokenization/chat formatting fixed. A native-target-tokenizer study would be separate.

### Candidate alignment and feature semantics

All audited training samples have `len(top_k_logits) == len(sequence_logprobs)+1`. The cache construction computes observed-token logprob `sequence_logprobs[t]` from next-token logits row `t`, so aligned rowwise comparisons use `top_k_logits[:-1]` against the entire sequence-logprob vector. The top-k entries are raw logits, not normalized full-vocabulary logprobs. A softmax over top512 yields a truncated/conditional distribution; subtracting its logprob from a full-vocabulary observed-token logprob is not an exact confidence gap.

The feature arrays also include the cached chat/template sequence; the live prompt loosely describes 'generated' claim tokens. Changing the interface description or trimming now would be a separate ablation. Preserve the historical main-study contract, record this limitation, and do not silently redefine inputs for selected candidates.

### Preserve frozen selection throughout analysis

The prepared screen and evolution phases all export `frozen_estimator.py`; keep screen winners separate from evolution winners. The screen has 30 requests and frozen contexts rather than 60 adaptive requests. Do not include screen runs as extra primary seeds. Do not repeatedly inspect OOD scores and then change prompts while describing the result as a fixed confirmatory protocol.

## Finishing criteria

Before stopping the paid pod, copy source configs, prompts, responses, ledgers, training/validation predictions, frozen sources, nine-dataset predictions/tables, environment/model metadata and timing records to durable local storage. Verify counts and hashes. Provider-level pod stop must actually occur; terminating a client process or shutting down an inference server alone does not release the paid GPU allocation.

The final report should state which planned jobs completed at full budget, which were partial/skipped, the actual GPU count/time, the strongest source-validation-selected OOD result, and the answer to each extension idea. Do not label merely prepared jobs as completed experiments.
