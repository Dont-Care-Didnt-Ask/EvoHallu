# Parent-selection mechanism

Status: **complete_three_seeds**. Both rules evaluated on each same eligible fitness-group pool observed in rank-policy runs; not a causal evolution-outcome comparison or a diversity guarantee.

Inherited softmax(score / recorded_temperature) versus recorded 0.8 * ascending_average_rank / sum(ranks) + 0.2 / n. Eligibility, island choice, and within-group shorter-source weighting remain inherited.

First-draw entropy, top/worst probability, and expected training fitness describe the first weighted draw. Two parents are sampled without replacement, then sorted; n<=2 always selects every eligible group. Exact unordered-set probabilities, set entropy/TV, group inclusion, and expected mean fitness describe this actual two-parent group selection.

Observed calls: 90; excluded one-group calls: 10; comparable calls: 80; pools with >2 eligible groups: 76. Recorded inherited temperatures: [40.0].

| Metric, mean across nondegenerate observed calls | Rank + 20% uniform | Inherited softmax |
| --- | ---: | ---: |
| First-draw normalized entropy | 0.95271 | 0.98257 |
| First-draw best-group probability | 0.36832 | 0.29514 |
| First-draw worst-group probability | 0.14043 | 0.20081 |
| Expected first-draw fitness | 69.77493 | 69.25613 |
| Expected mean fitness of two selected groups | 69.28926 | 68.83441 |
| Best-group inclusion probability in two-parent set | 0.65938 | 0.56221 |
| Worst-group inclusion probability in two-parent set | 0.33091 | 0.42936 |
| Two-parent set normalized entropy (>2 eligible groups) | 0.93094 | 0.97884 |

Mean total-variation distance: 0.09394.

Rank weighting can increase or decrease exploitation relative to temperature 40 depending on score gaps; use the signed differences above rather than assuming the direction.

Directed training AUROC percentage points. No validation or test outcomes used.

All nondegenerate recorded events contribute to descriptive means, including partial runs. One-fitness-group events excluded. Primary means weight seeds equally and require all three full 60-request runs plus 30 events each.

Per-seed summaries, exact two-parent metrics, and input hashes are preserved in JSON; per-event metrics are in CSV.

Mean two-parent set total-variation distance: 0.12637.

P({i,j}) = p_i*p_j/(1-p_i) + p_j*p_i/(1-p_j), i<j; inclusion_i = sum of sets containing i; expected parent mean = sum P({i,j})*(score_i+score_j)/2. Set normalized entropy undefined for n<=2 (only one possible set).

All three seeds complete. Equal-seed mean changes: expected first-draw fitness +0.52096, expected two-parent mean fitness +0.45626, first-draw normalized entropy -0.02988.
