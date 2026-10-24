# Temporal evolution diversity

Fixed 512-row training probe; no validation/test predictions or labels read.

Training-useful means training fitness at least the NLL baseline; this is exploratory and reuses the training objective.

Primary averages require all three complete seeds. Partial runs are shown below and never enter those averages.

| Run | Status | Rounds | Requests | Training-valid | Nonconstant | Near baseline | Near previous | Novel | Novel + training-useful | Abs-novel |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| next_evo_original_2107 | full_budget | 1-10 | 20 | 8 | 8 | 3 | 3 | 4 | 2 | 4 |
| next_evo_original_2107 | full_budget | 11-20 | 20 | 9 | 9 | 0 | 1 | 8 | 1 | 8 |
| next_evo_original_2107 | full_budget | 21-30 | 20 | 13 | 13 | 7 | 7 | 6 | 0 | 6 |
| next_evo_neutral_2107 | full_budget | 1-10 | 20 | 13 | 12 | 0 | 2 | 10 | 0 | 9 |
| next_evo_neutral_2107 | full_budget | 11-20 | 20 | 4 | 4 | 0 | 3 | 1 | 0 | 1 |
| next_evo_neutral_2107 | full_budget | 21-30 | 20 | 12 | 11 | 0 | 5 | 6 | 0 | 6 |
| next_evo_diverse_2107 | full_budget | 1-10 | 20 | 15 | 15 | 1 | 4 | 10 | 1 | 10 |
| next_evo_diverse_2107 | full_budget | 11-20 | 20 | 14 | 11 | 3 | 5 | 6 | 0 | 6 |
| next_evo_diverse_2107 | full_budget | 21-30 | 20 | 12 | 10 | 1 | 5 | 5 | 0 | 5 |
| next_evo_original_4319 | full_budget | 1-10 | 20 | 11 | 11 | 0 | 2 | 9 | 1 | 9 |
| next_evo_original_4319 | full_budget | 11-20 | 20 | 14 | 13 | 4 | 6 | 6 | 1 | 6 |
| next_evo_original_4319 | full_budget | 21-30 | 20 | 11 | 11 | 2 | 4 | 7 | 1 | 7 |
| next_evo_neutral_4319 | full_budget | 1-10 | 20 | 11 | 11 | 1 | 6 | 4 | 0 | 4 |
| next_evo_neutral_4319 | full_budget | 11-20 | 20 | 14 | 13 | 1 | 10 | 3 | 0 | 1 |
| next_evo_neutral_4319 | full_budget | 21-30 | 20 | 11 | 11 | 1 | 4 | 7 | 0 | 7 |
| next_evo_diverse_4319 | full_budget | 1-10 | 20 | 9 | 8 | 1 | 2 | 5 | 0 | 5 |
| next_evo_diverse_4319 | full_budget | 11-20 | 20 | 9 | 7 | 0 | 1 | 6 | 0 | 6 |
| next_evo_diverse_4319 | full_budget | 21-30 | 20 | 7 | 7 | 0 | 3 | 4 | 1 | 3 |
| next_evo_original_8741 | full_budget | 1-10 | 20 | 14 | 14 | 0 | 0 | 14 | 2 | 14 |
| next_evo_original_8741 | full_budget | 11-20 | 20 | 8 | 8 | 2 | 1 | 6 | 1 | 6 |
| next_evo_original_8741 | full_budget | 21-30 | 20 | 13 | 13 | 3 | 3 | 10 | 2 | 9 |
| next_evo_neutral_8741 | full_budget | 1-10 | 20 | 13 | 13 | 2 | 3 | 9 | 0 | 7 |
| next_evo_neutral_8741 | full_budget | 11-20 | 20 | 8 | 7 | 1 | 5 | 2 | 0 | 2 |
| next_evo_neutral_8741 | full_budget | 21-30 | 20 | 12 | 9 | 0 | 7 | 2 | 0 | 2 |
| next_evo_diverse_8741 | full_budget | 1-10 | 20 | 14 | 13 | 1 | 1 | 11 | 0 | 11 |
| next_evo_diverse_8741 | full_budget | 11-20 | 20 | 11 | 8 | 1 | 4 | 4 | 2 | 4 |
| next_evo_diverse_8741 | full_budget | 21-30 | 20 | 12 | 11 | 0 | 2 | 9 | 0 | 9 |
| next_evo_compositional_2107 | full_budget | 1-10 | 20 | 6 | 6 | 0 | 0 | 6 | 0 | 6 |
| next_evo_compositional_2107 | full_budget | 11-20 | 20 | 9 | 9 | 1 | 0 | 8 | 0 | 8 |
| next_evo_compositional_2107 | full_budget | 21-30 | 20 | 12 | 12 | 0 | 5 | 7 | 3 | 7 |
| next_evo_compositional_4319 | full_budget | 1-10 | 20 | 7 | 6 | 0 | 0 | 6 | 0 | 6 |
| next_evo_compositional_4319 | full_budget | 11-20 | 20 | 11 | 11 | 1 | 2 | 8 | 1 | 8 |
| next_evo_compositional_4319 | full_budget | 21-30 | 20 | 11 | 11 | 0 | 3 | 8 | 1 | 8 |
| next_evo_compositional_8741 | full_budget | 1-10 | 20 | 7 | 7 | 0 | 1 | 6 | 1 | 6 |
| next_evo_compositional_8741 | full_budget | 11-20 | 20 | 7 | 7 | 0 | 2 | 5 | 0 | 5 |
| next_evo_compositional_8741 | full_budget | 21-30 | 20 | 8 | 8 | 1 | 2 | 5 | 1 | 5 |
| next_evo_neutral_rank_uniform_2107 | full_budget | 1-10 | 20 | 12 | 10 | 0 | 4 | 6 | 0 | 6 |
| next_evo_neutral_rank_uniform_2107 | full_budget | 11-20 | 20 | 13 | 13 | 1 | 6 | 6 | 0 | 5 |
| next_evo_neutral_rank_uniform_2107 | full_budget | 21-30 | 20 | 10 | 10 | 5 | 6 | 4 | 0 | 4 |
| next_evo_neutral_rank_uniform_4319 | full_budget | 1-10 | 20 | 9 | 8 | 1 | 3 | 4 | 1 | 4 |
| next_evo_neutral_rank_uniform_4319 | full_budget | 11-20 | 20 | 12 | 12 | 2 | 10 | 2 | 0 | 2 |
| next_evo_neutral_rank_uniform_4319 | full_budget | 21-30 | 20 | 12 | 11 | 0 | 6 | 5 | 0 | 5 |
| next_evo_neutral_rank_uniform_8741 | full_budget | 1-10 | 20 | 11 | 11 | 0 | 6 | 5 | 0 | 5 |
| next_evo_neutral_rank_uniform_8741 | full_budget | 11-20 | 20 | 14 | 14 | 3 | 6 | 7 | 0 | 5 |
| next_evo_neutral_rank_uniform_8741 | full_budget | 21-30 | 20 | 7 | 7 | 2 | 5 | 2 | 0 | 2 |
| next_evo_neutral_interface_2107 | full_budget | 1-10 | 20 | 5 | 5 | 0 | 1 | 4 | 0 | 4 |
| next_evo_neutral_interface_2107 | full_budget | 11-20 | 20 | 14 | 13 | 0 | 5 | 8 | 0 | 6 |
| next_evo_neutral_interface_2107 | full_budget | 21-30 | 20 | 10 | 10 | 0 | 6 | 4 | 0 | 3 |

## Complete three-seed averages

- Qwen/Qwen3.5-9B / evolution_screen / compositional / inherited:
  - Rounds 1-10: mean novel count 6.00, mean novel + training-useful 0.33; all three seeds complete.
  - Rounds 11-20: mean novel count 7.00, mean novel + training-useful 0.33; all three seeds complete.
  - Rounds 21-30: mean novel count 6.67, mean novel + training-useful 1.67; all three seeds complete.
- Qwen/Qwen3.5-9B / evolution_screen / diverse / inherited:
  - Rounds 1-10: mean novel count 8.67, mean novel + training-useful 0.33; all three seeds complete.
  - Rounds 11-20: mean novel count 5.33, mean novel + training-useful 0.67; all three seeds complete.
  - Rounds 21-30: mean novel count 6.00, mean novel + training-useful 0.33; all three seeds complete.
- Qwen/Qwen3.5-9B / evolution_screen / neutral / inherited:
  - Rounds 1-10: mean novel count 7.67, mean novel + training-useful 0.00; all three seeds complete.
  - Rounds 11-20: mean novel count 2.00, mean novel + training-useful 0.00; all three seeds complete.
  - Rounds 21-30: mean novel count 5.00, mean novel + training-useful 0.00; all three seeds complete.
- Qwen/Qwen3.5-9B / evolution_screen / original / inherited:
  - Rounds 1-10: mean novel count 9.00, mean novel + training-useful 1.67; all three seeds complete.
  - Rounds 11-20: mean novel count 6.67, mean novel + training-useful 1.00; all three seeds complete.
  - Rounds 21-30: mean novel count 7.67, mean novel + training-useful 1.00; all three seeds complete.
- Qwen/Qwen3.5-9B / sampler_control / neutral / rank_uniform:
  - Rounds 1-10: mean novel count 5.00, mean novel + training-useful 0.33; all three seeds complete.
  - Rounds 11-20: mean novel count 5.00, mean novel + training-useful 0.00; all three seeds complete.
  - Rounds 21-30: mean novel count 3.67, mean novel + training-useful 0.00; all three seeds complete.

The previous-candidate archive grows over time. A decline in novelty alone does not establish collapse; compare baseline and sibling similarity too.

Hashes, absolute-correlation supplements, sibling similarity, and per-candidate diagnostics are preserved in JSON/CSV.
