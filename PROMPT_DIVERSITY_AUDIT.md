# EvoHallu: Prompt Diversity Audit

21 October 2026. Retrospective analysis of saved prompts and predictions. No GPU use, model requests, candidate execution, or changes to the experimental pipeline.

**Follow-up:** runnable ablations, including a multi-feature compositional arm inspired by the Claude example supplied by the user, are now described in [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md). The author also clarified paper-compatible per-dataset `max(AUC,100-AUC)` reporting; the NLL baseline is 65.64 directed / 68.71 paper-compatible. Labels and the retrospective diversity measurements below are unchanged.

**Finding:** the current prompt explicitly restricts exploration, and saved outputs show substantial repetition within shared-prompt batches. Prompt anchoring is a credible bottleneck, especially for Laguna and Qwen27B. Its causal contribution and its effect on final AUROC have not yet been measured by a prompt ablation.

## What the prompt actually asks for

The active `system_prompt_fabric()` in [task_uncertainty_quantification.py](src/packing/evaluate/uncertainty_quantification/task_uncertainty_quantification.py) says:

```text
**EXPECTED STRUCTURE**
- Build off a strong baseline: {baseline}
- For additional signal, use {idea}
...
Ideate a novel detector, based on EXPECTED STRUCTURE ...
```

The baseline is sampled from only three choices: Sequence NLL, LogProbSlope, and PositionWeightedGap, with configured weights **40%, 50%, 10%**. An additional idea is sampled from 12 choices, such as standard deviation, positional weighting, windows, and extreme-value aggregation. This permits useful local search but makes baseline-plus-adjustment the default architecture for every proposal. Asking for a “novel detector” does not remove that restriction.

The prescription was already present in the original repository import. Our campaign corrected the description of `sequence_logprobs` from “logits” to “log-probabilities”; it retained the baseline-plus-signal instructions.

There are four additional implementation details:

1. **Both siblings receive identical messages.** [extension_trial.py](src/experiments/extension_trial.py) constructs one chat per round and sends it twice, changing only the request seed. Neither call receives a separate role or method family. Different seeds permit stochastic variation; they do not require different approaches.
2. **Parent examples provide a second anchor.** The initial population is NLL. Later prompts include complete parent programs, their comments/docstrings, and training scores. Across the primary histories, user-message length was a median **4.17 times** system-message length, measured in characters. This measures context volume, not psychological influence.
3. **The independent control retains the same system prescription.** It removes parent examples but still calls `system_prompt_fabric()`. It tests generation without evolutionary examples, not generation without baseline guidance.
4. **The unused structured mode still assigns its family per round.** Both siblings would share it. It also appends a family requirement to the existing baseline/auxiliary prescription, potentially restricting or complicating the requested design. No completed structured-mode result supports a diversity claim.

Across **201 primary rounds**, only **16 distinct system messages** occurred. Realized baseline exposure was **118 NLL / 63 slope / 20 gap** rounds, or **58.7% / 31.3% / 10.0%**. These differ from the configured weights because these are short seeded schedules, heavily reused across writers. For example, seed 1047 gives 18/9/3 exposures over 30 rounds; seed 1048 gives 17/10/3. Reusing schedules helps matched writer comparisons, but those repeated exposures are not independent prompt draws.

## What the saved predictions show

Coverage: **7 primary evolution runs, 402 requests, 283 valid candidates**. Of these, 280 have nonconstant saved validation predictions. A near-identical pair means **positive Spearman correlation ≥0.995** between its saved prediction rankings over the same 1,402 PopQA validation examples. Invalid and constant-output candidates are excluded from pair comparisons.

“Siblings” are the two candidates generated in one round. The comparison group contains valid candidate pairs from different rounds separated by one or two rounds, always within the same run.

| Writer | Near-identical sibling pairs | Near-identical nearby-round pairs |
| --- | ---: | ---: |
| Laguna XS 2.1 | **12/34 (35.3%)** | 26/236 (11.0%) |
| Qwen3.8-27B | **8/50 (16.0%)** | 12/382 (3.1%) |
| Qwen3.5-9B | **2/23 (8.7%)** | 3/132 (2.3%) |
| Qwen3.5-4B | 0/4 | 0/18 |
| **Total** | **22/111 (19.8%)** | **41/768 (5.3%)** |

The 4B sample is too small, with too much invalidity, to establish healthy diversity. High diversity among incorrect programs is not the objective.

All **283 valid source hashes are different**, yet **six sibling pairs have exactly the same saved rank ordering**, including ties. Source-code deduplication misses these repeated behaviours. Also, **72/280 nonconstant candidates (25.7%)** have correlation ≥0.995 with the specific named baseline in their own system prompt. These are different statistics from the earlier nearest-previous-candidate audit.

There is supporting evidence without adaptive parent selection. In the two fixed-NLL-example controls, non-sibling candidates receiving the **same system text** have **12/76 near-identical pairs (15.8%)**, versus **30/1,319 (2.3%)** when the system text differs. The parent example stays fixed in these controls. This strengthens the prompt hypothesis, though the comparison remains observational and uses few repeated prompt variants.

**Limits:** pair observations share candidates and are statistically dependent. Validity filtering changes which pairs can be compared. Nearby rounds can have different parents, population states, and random seeds. These are descriptive rates, not independent-trial significance tests or a measured causal effect of replacing the prompt. Correlation close to one also does not prove complete functional equivalence or rule out a useful small improvement.

## Some apparent novelty also disappears in implementation

A concrete example is **Qwen27B, seed 1048, round 14, candidate index 1**. It proposes LogProbSlope plus a bottom-20% confidence-margin feature, but its code contains:

```python
n_tokens = len(logprobs)
if top_k_logits.shape[0] == n_tokens:
    # compute margin_component
    ...
else:
    margin_component = 0.0
score = slope_score + 0.1 * margin_component
```

The documented inputs have `n_i` top-k rows and `n_i - 1` sequence log-probabilities. Under that contract, the equality check fails and the extra signal becomes zero. Its saved validation ranking is exactly the supplied LogProbSlope ranking, consistent with this failure mode. Evidence: the saved candidate ledger in `out/logs/mod1/uncertainty_quantification_qwen38_evo_1048/` and the reference vectors in `out/campaign/original_validation/stratified_val_scores.npz`.

Therefore, a repeated detector can arise from a restrictive design instruction, repeated parent examples, writer preferences, or a disabled extra feature. More elaborate generated descriptions alone would not resolve all of these mechanisms. The next interface should include a verified alignment example and smoke checks for active feature branches.

## Recommended next experiment

First isolate the prompt effect with **frozen parent contexts**, rather than changing the sampler, wrapper, temperature, and prompts simultaneously:

| Arm | Change | Question |
| --- | --- | --- |
| A | Existing prompt and shared sibling messages | Reference behaviour |
| B | Replace the mandatory baseline-plus-signal block with open formula choice; remove the later reference to EXPECTED STRUCTURE | Does removing the explicit architecture prescription increase useful diversity? |
| C | B plus distinct, balanced family assignments for each candidate in a round | Does deliberate sibling diversification add value? |
| D | B plus guidance to combine several complementary feature families, with shared sibling messages | Does explicit compositional design help? |

Keep the same parent code/scores, input contract, writer, token cap, temperature 0.7, top-p 0.8, top-k 20, and paired request seeds. Retain the current interface wording in all arms so this screen changes the formula guidance rather than introducing a simultaneous wrapper fix. Choose the contexts uniformly across saved rounds before examining their outcomes. Assign families per **candidate**, not just per round.

For B, a neutral replacement is:

```text
Choose a deterministic scoring formula using the available inputs.
The signal family, aggregation principle, and formula structure are open.
Use the shown training results as evidence when choosing your approach.
```

Also replace response-format step 2 with “Propose a detector and briefly state its signal and aggregation principle.” Keep the input shapes, output requirements, and remaining response format unchanged.

For C, rotate assignments across confidence gaps, local surprisal patterns, distribution concentration, positional effects, and robust aggregation. Preserve an exploitation role as well as exploration; forced novelty can discard useful baseline refinements. A completely example-free exploration arm would be a separate later test, because removing parents changes another factor.

A cheap screen is **15 frozen contexts × 4 arms × 2 candidates = 120 writer requests** with Qwen3.5-9B. At the earlier recorded end-to-end trial rate, this is approximately **40 minutes including generation and scoring**, plus serving/setup overhead and possible prompt-length differences. The prepared single-worker evaluation configuration may take longer. This is a projection, not a completed run or resource reservation. It measures conditional proposal behaviour, not the full long-term evolution effect. D follows the multi-feature Claude example supplied by the user; it imposes no feature-count ceiling or new complexity penalty.

Measure validity, distinct **training-probe** rank patterns, sibling correlation, generation tokens, and best training AUROC per fixed request budget. Use no validation predictions in prompts, parent novelty, or adaptive generation. If the screen improves diversity while preserving quality and validity, follow with equal-budget evolution runs over at least three fresh seeds, using a predeclared validation selection rule and frozen final evaluation. Previously inspected validation/test data provide exploratory rather than independently confirmed quality results.

**Decision:** prompt restriction is a real issue worth testing, and the existing records provide supporting evidence. They do not establish that diversity tuning will improve final detector AUROC. No new macro AUROC result was produced by this audit; the existing `is_hallu` convention and **65.64** macro-AUROC NLL baseline remain unchanged.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_prompt_diversity.py
```

The script reads existing records and arrays, parses source with AST, and writes [prompt_diversity_audit.json](results/prompt_diversity_audit.json). It never imports or executes candidate programs. Static input-access patterns in that JSON are literal-code-access diagnostics, not verified signal-family classifications. The JSON includes per-run summaries, prompt metadata, per-candidate diagnostics, and individual valid/nonconstant sibling comparisons. Future search-time novelty must use training/probe predictions rather than the validation arrays used for this retrospective audit.
