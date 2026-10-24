# GLM final selected-program inspection

Both **GLM-4.7-Flash seed-2107** jobs completed their full **60-request** budgets with `round_budget` termination. The CPU driver froze both primaries before this inspection. Writer revision: `7dd20894a642a0aa287e9827cb1a1f7f91386b67`. Fixed source features remain **Llama3.1-8b-instruct**; GLM writes detector code rather than generating claims or supplying their probabilities.

| Prompt policy | Valid / attempted | Selected candidate | Directed PopQA train / validation AUROC | Frozen source |
| --- | ---: | --- | ---: | --- |
| Original | 28 /60 | `4d80eca6979e…` | 79.8256 /78.7667 | [Original program](../../out/logs/research20261021/uncertainty_quantification_nextglm_evo_original_2107/frozen_estimator.py) |
| Diverse | 36 /60 | canonical NLL fallback | 74.3286 /75.7504 | [Baseline program](../../out/logs/research20261021/uncertainty_quantification_nextglm_evo_diverse_2107/frozen_estimator.py) |

There are **120 requests, 64 pipeline-valid candidates and 208,611 reported tokens** in this pair. Recorded elapsed times are 3,182.67 and 2,934.23 seconds, including CPU evaluation/shared serving; they are not additive GPU-hours. Selection uses directed PopQA validation with the existing baseline fallback, without benchmark/target scores. Diverse was the prospectively chosen intervention by mean validation across all three 9B discovery seeds, among neutral/diverse/compositional; original remains the reference.

## Original: a familiar single correlation

The actual score is

`S(lp) = float32(−PearsonCorr([0,1,…,n−1], array(lp,float32)))`.

The program calls `np.corrcoef` for each claim independently. Its variable is named `slope`, but it is a **Pearson correlation**, not an ordinary least-squares slope. It has one effective signal: whether observed-token logprob tends to decrease with position. It contains no NLL-sum contribution, local-window statistic, top-k term, learned coefficient, batch normalization or cross-claim operation. The enumerated sample index is unused; no advertised extra signal is hidden in a conditional branch.

This rediscovers the inherited [LogProbSlope](../../src/packing/evaluate/uncertainty_quantification/found_formulas.py#L649) mathematical score. Promotion of stored float16 LPs to float32 preserves their values exactly; `np.corrcoef` defaults to float64 covariance/correlation before the result is cast to float32. For finite, nonempty recorded vectors on which the inherited baseline's guards do not trigger, both implementations construct the same float64 correlation inputs. This is a static conditional equivalence, **not a newly established all-input executable alias or a new precision-only NLL variant**.

There is one material implementation difference: the GLM code **omits** the inherited empty/`np.allclose`-constant guards. Empty, one-token or constant vectors may produce NaN; near-constant vectors can also differ from the guarded baseline. No guard was added and no new failure test or scoring was run for this inspection. The comment about confidence degradation is a hypothesis; cached positions include template tokens and do not isolate a causal later-claim mechanism.

Original source SHA256: `869e426b9002b5c7d3ae1d4ad4bee2131a09f86e8bb0d6f0e8b9e103aaec4c75`.

## Diverse: the unchanged NLL fallback

The final selected source computes `array([-sum(lp) for lp in sequences], dtype=float32)`. The sum uses the cached array's native **float16** accumulation before output casting. This is the exact canonical fallback, SHA256 `2441fedd58e6043981283df3fc50a87c6e5a223090439c2b9996472eabf9af7d`, rather than a generated extra feature or float32-sum variant. It is pointwise and has no top-k dependency or batch normalization. Generated shortlist entries tying the baseline do not replace the selected fallback.

## Provenance and interpretation

All **14 manifest source pins** match. Each runner-frozen source and staged primary matches its registry SHA byte-for-byte; final summaries agree with primary candidate identity, validation score, full 60/60 budget and writer. The exact pilot [manifest](../next_experiments/glm47_seed2107_manifest.json) has SHA256 `e28f72a01f11b16bf2176b958b678f720353295e44f637d2903014b96f05c902`. Registry hashes are timestamped snapshots because other separately managed cohorts may be staged later.

This is a **single-seed selected-program inspection**. It does not imply that every generated GLM program was a baseline alias, establish a general writer-family ranking, or explain prompt-policy effects. Additional prepared seeds are not assumed executed; final scheduling/coverage receipts govern those separately. [JSON evidence](GLM_CANDIDATE_INSPECTION.json) retains hashes, final identities, summary/ledger provenance, train/validation values, active signals, precision and guard caveats. No generated program was executed, repaired, rescored or promoted to write these notes, and no primary registry or execution manifest was changed.
