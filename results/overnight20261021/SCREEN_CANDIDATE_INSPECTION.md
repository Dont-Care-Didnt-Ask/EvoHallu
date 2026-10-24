# Prompt-screen candidate inspection

Snapshot: **2026-10-21T19:06:47.966466+00:00**. **All four screens have finished: 30 requests each, 120 total.** This supersedes the earlier partial ordering in [initial inspection](CANDIDATE_INSPECTION.md).

Writer: **Qwen3.5-9B**. Fixed feature model: **Llama3.1-8b-instruct**. Each screen uses the same 15 frozen parent contexts and two seeded siblings; these are screening jobs, not 30-round adaptive evolutionary searches. Numbers are directed PopQA train/validation AUROC on 5,605/1,402 rows. NLL baseline: **74.3286 / 75.7504**. No held-out test predictions were read.

## Validation leaders

| Policy | Valid / evaluated requests | Train AUROC | Validation AUROC | Gain vs validation NLL | Exact source |
| --- | ---: | ---: | ---: | ---: | --- |
| original | 19/30 | 79.3704 | 77.7002 | +1.9498 pp | [699633365cbf](screen_candidate_sources/original_rank1_699633365cbf.py) |
| neutral | 19/30 | 80.7778 | 80.4055 | +4.6551 pp | [0a747b101a0b](screen_candidate_sources/neutral_rank1_0a747b101a0b.py) |
| diverse | 23/30 | 79.3948 | 79.5130 | +3.7626 pp | [4eb5b0928588](screen_candidate_sources/diverse_rank1_4eb5b0928588.py) |
| compositional | 11/30 | 79.6150 | 80.2857 | +4.5353 pp | [ae79b1db3f05](screen_candidate_sources/compositional_rank1_ae79b1db3f05.py) |

Ranks use validation only. Comparing the largest value from each screen does **not** establish a prompt-policy effect. The separate matched-context diversity analysis uses all requests, including failures.

## What these programs actually compute

Write `l[t]` for cached observed-token log probability, `N=−sum(l)`, `w[t]=t/(n−1)`, and `g[t]=log(q_top1[t])−l[t]`, with `q` normalized over the **top 512 logits only**.

- **Original leader:** `[0.7·(−corr(position,l)) + 0.3/(1+range(l[last 10%]))] / (mean(abs(l))+1e−8)`. Two feature groups: confidence trend and inverse tail range, followed by global surprise normalization. This extends `LogProbSlope`; it is not exactly that baseline. A narrow range is not itself low confidence, and the cached tail includes template tokens.
- **Neutral leader:** `N + 4·sum(w·g) + 2·Var(g[last 25%])`. Three feature groups: sequence surprise, later-position gap accumulation and tail instability. Both gap-based terms are active; the program is not an NLL alias.
- **Diverse leader:** `N + 2·(mean(g[last quarter])−mean(g[first quarter])) / max(mean(g[first quarter]),1e−6) + 2.5·Var(g[last quarter])`. Three groups: sequence surprise, relative early/late gap drift and tail instability. The small denominator floor can produce large values if the early gap is near zero; this is a numerical sensitivity, not an observed failure of this valid candidate.
- **Compositional leader:** `0.4·Var(−l[last 10%]) + 0.3·mean(−l[last 20%]) + 3·[−corr(position,l)]`. Three feature groups: tail variance, tail mean surprise and global confidence trend. It uses sequence log probabilities only; there is no top-k dependency or NLL-sum term. As with the original leader, `pearsonr` can return NaN without raising an exception on a constant sequence; saved training/validation outputs are finite.

## Complexity and baseline similarity

Complexity counts describe the actual code after removing docstrings: AST nodes and lines produced by `ast.unparse`. They are not feature counts or a complexity penalty. Correlations are Spearman correlations of saved predictions.

| Leader | Lines / AST nodes | NLL correlation, all training rows | NLL correlation, validation | LogProbSlope correlation, validation |
| --- | ---: | ---: | ---: | ---: |
| original | 40 / 340 | 0.520591 | 0.520680 | 0.954294 |
| neutral | 49 / 439 | 0.773963 | 0.792922 | 0.872428 |
| diverse | 58 / 517 | 0.700853 | 0.708590 | 0.762005 |
| compositional | 44 / 352 | 0.635654 | 0.637733 | 0.924611 |

None of these four leaders is an exact NLL or LogProbSlope alias. Original’s third-ranked candidate is almost NLL (`rho=0.9999926` on training): its rank statistic adds approximately a constant offset. Another original candidate, [03b82ca118be](screen_candidate_sources/original_exact_nll_alias_03b82ca118be.py), is **exactly NLL on all 5,605 training rows**: it looks up the top-1 ID inside the same top-k row, necessarily obtaining rank zero.

## Other top validation candidates

| Policy | Rank 2: code / validation AUROC | Rank 3: code / validation AUROC |
| --- | --- | --- |
| original | [8645811f2d67](screen_candidate_sources/original_rank2_8645811f2d67.py) / 76.0421 | [838acd05eba9](screen_candidate_sources/original_rank3_838acd05eba9.py) / 75.7545 |
| neutral | [f08d667084de](screen_candidate_sources/neutral_rank2_f08d667084de.py) / 79.7566 | [385728e4b958](screen_candidate_sources/neutral_rank3_385728e4b958.py) / 78.9083 |
| diverse | [3fceb33bdfb6](screen_candidate_sources/diverse_rank2_3fceb33bdfb6.py) / 77.2738 | [208f17d7d0b2](screen_candidate_sources/diverse_rank3_208f17d7d0b2.py) / 76.7750 |
| compositional | [3e8affe01279](screen_candidate_sources/compositional_rank2_3e8affe01279.py) / 78.3886 | [b7b466e3a44f](screen_candidate_sources/compositional_rank3_b7b466e3a44f.py) / 76.5734 |

Neutral rank 2 adds tail weighted variance divided by head mean absolute log probability. Neutral rank 3 adds gap moments, but its advertised “weighted variance” uses squared weights and can be negative; it is more accurately a second-moment contrast. Diverse rank 2 adds an OLS gap slope and gap variance; diverse rank 3 adds later-weighted log-probability standard deviation.

Compositional rank 2 is `N + 2.5·mean(g) + 2·(mean(g[last quarter])−mean(g[first quarter]))`. Its rank 3 is `0.5·N + Var(l)/(n+1e−8) + 2·mean_t(max(top512_logits[t])−mean(top512_logits[t]))`: a three-group combination of surprise, length-normalized variation and raw-logit spread. The max-minus-mean spread is shift-invariant, but is not entropy or total probability mass.

A lower-ranked compositional program, [ea353158e423](screen_candidate_sources/compositional_dead_position_gate_ea353158e423.py), declares four terms but has an inactive position term: `n >= int(n/0.1)+int(n/0.1)` is impossible for positive `n`. This does not affect the final compositional winner, which is a different program.

## Numerical interpretation and provenance

Gap candidates normalize inside the top 512 logits: their gap equals the full-vocabulary top-1 gap minus log(top-512 probability mass), up to stored numerical precision. This remains a usable feature, but it is not the exact full-vocabulary probability gap. `logits[t]` correctly predicts the token associated with `l[t]`; the gap leaders leave the final extra top-k row unused. No `n==n+1` shape guard disables their additions.

No code was repaired, refit, shortened or penalized. Training/validation outputs were already finite for these valid candidates; the notes identify interpretation limits and possible sensitivities, not new test failures. Full candidate hashes, exact exported-file SHA256 hashes, ledger snapshot hashes, complexity counts and train/probe/validation correlations are retained in [inspection data](SCREEN_CANDIDATE_INSPECTION.json). Exported Python files preserve generated source exactly.
