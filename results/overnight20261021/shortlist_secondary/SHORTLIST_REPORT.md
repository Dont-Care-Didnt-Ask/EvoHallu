# Validation-shortlist secondary analysis

Status: **incomplete_secondary_shortlist**.

Top-three means average individual method AUROCs, not predictions or ensembles. No test-best or oracle selection.

| Policy | Validation rank | Seeds | Directed mean ± SD | Folded mean ± SD | Duplicate-disjoint folded |
| --- | ---: | --- | ---: | ---: | ---: |
| original | 1 | [2107, 4319, 8741] | 67.823 ± 1.409 | 69.943 ± 1.059 | 69.879 ± 1.044 |
| original | 2 | [2107, 4319, 8741] | 66.916 ± 1.396 | 69.663 ± 0.881 | 69.619 ± 0.852 |
| original | 3 | [2107, 4319, 8741] | PENDING | PENDING | PENDING |
| neutral | 1 | [2107, 4319, 8741] | 65.644 ± 0.000 | 68.711 ± 0.000 | 68.685 ± 0.000 |
| neutral | 2 | [2107, 4319, 8741] | 65.554 ± 0.157 | 68.517 ± 0.336 | 68.488 ± 0.341 |
| neutral | 3 | [2107, 4319, 8741] | 65.258 ± 0.668 | 67.325 ± 2.400 | 67.278 ± 2.437 |
| diverse | 1 | [2107, 4319, 8741] | 67.748 ± 2.107 | 70.355 ± 1.648 | 70.307 ± 1.627 |
| diverse | 2 | [2107, 4319, 8741] | 65.644 ± 0.000 | 68.711 ± 0.000 | 68.685 ± 0.000 |
| diverse | 3 | [2107, 4319, 8741] | 65.644 ± 0.000 | 68.711 ± 0.000 | 68.685 ± 0.000 |
| compositional | 1 | [2107, 4319, 8741] | 68.297 ± 1.550 | 70.002 ± 0.911 | 69.921 ± 0.878 |
| compositional | 2 | [2107, 4319, 8741] | 66.699 ± 1.828 | 69.342 ± 1.092 | 69.292 ± 1.052 |
| compositional | 3 | [2107, 4319, 8741] | 66.684 ± 1.801 | 69.323 ± 1.060 | 69.274 ± 1.020 |

Per-run frozen ranks (no test-based ordering):

| Run | Validation rank | Validation AUROC | Directed nine | Folded nine | Strict folded nine |
| --- | ---: | ---: | ---: | ---: | ---: |
| next_evo_original_2107 | 1 | 76.9672 | 66.308 | 69.344 | 69.318 |
| next_evo_original_2107 | 2 | 76.8033 | 66.275 | 69.283 | 69.253 |
| next_evo_original_2107 | 3 | 76.1889 | 65.674 | 68.730 | 68.703 |
| next_evo_neutral_2107 | 1 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_neutral_2107 | 2 | 75.0156 | 65.373 | 68.129 | 68.095 |
| next_evo_neutral_2107 | 3 | 69.9220 | 64.487 | 64.554 | 64.464 |
| next_evo_diverse_2107 | 1 | 76.8771 | 67.744 | 70.345 | 70.298 |
| next_evo_diverse_2107 | 2 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_diverse_2107 | 3 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_original_4319 | 1 | 79.2061 | 69.095 | 71.167 | 71.083 |
| next_evo_original_4319 | 2 | 78.7959 | 68.517 | 70.671 | 70.593 |
| next_evo_original_4319 | 3 | 78.7667 | 68.558 | 70.657 | 70.579 |
| next_evo_neutral_4319 | 1 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_neutral_4319 | 2 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_neutral_4319 | 3 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_diverse_4319 | 1 | 79.5207 | 69.858 | 72.008 | 71.939 |
| next_evo_diverse_4319 | 2 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_diverse_4319 | 3 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_original_8741 | 1 | 78.4540 | 68.067 | 69.320 | 69.236 |
| next_evo_original_8741 | 2 | 76.1852 | 65.955 | 69.036 | 69.011 |
| next_evo_original_8741 | 3 | 76.0972 | PENDING | PENDING | PENDING |
| next_evo_neutral_8741 | 1 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_neutral_8741 | 2 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_neutral_8741 | 3 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_diverse_8741 | 1 | 75.7622 | 65.643 | 68.712 | 68.686 |
| next_evo_diverse_8741 | 2 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_diverse_8741 | 3 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_compositional_2107 | 1 | 76.8237 | 69.075 | 70.677 | 70.583 |
| next_evo_compositional_2107 | 2 | 76.5968 | 68.810 | 70.603 | 70.507 |
| next_evo_compositional_2107 | 3 | 76.5376 | 68.764 | 70.547 | 70.451 |
| next_evo_compositional_4319 | 1 | 76.6394 | 69.304 | 70.363 | 70.255 |
| next_evo_compositional_4319 | 2 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_compositional_4319 | 3 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_compositional_8741 | 1 | 78.2742 | 66.512 | 68.965 | 68.926 |
| next_evo_compositional_8741 | 2 | 75.7504 | 65.644 | 68.711 | 68.685 |
| next_evo_compositional_8741 | 3 | 75.7504 | 65.644 | 68.711 | 68.685 |

Top-three collection means average individual method AUROCs; they are not ensembles:

| Policy | Directed collection mean ± SD across seeds | Folded collection mean ± SD across seeds |
| --- | ---: | ---: |
| original | PENDING | PENDING |
| neutral | 65.485 ± 0.275 | 68.185 ± 0.912 |
| diverse | 66.345 ± 0.702 | 69.259 ± 0.549 |
| compositional | 67.227 ± 1.508 | 69.556 ± 0.941 |

All per-run validation scores, nine-dataset scores, strict scores, identities and hashes are in [JSON](SHORTLIST_REPORT.json).

The [source gallery](SOURCE_GALLERY.md) includes every rank slot, including baseline/duplicate aliases and missing valid candidates.

Exploratory retained collection; prior benchmark exposure, inherited split duplicates, no new blind test. Intervals condition on frozen programs.
