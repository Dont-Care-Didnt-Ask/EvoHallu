# Frozen-screen component ablations: scientific interpretation

Audited the completed remote report dated **2026-10-21 19:52:23 UTC**. Its JSON SHA256 is `5829a4f53289c1baf5dc7b84e24ed85bc7239a781424ef64e61b80dea176a6e2`. This audit only reads saved results and source; it launches no inference and changes no program or protocol.

**Integrity passed.** All 15 programs plus NLL were scored on all nine source datasets and the prescribed six target datasets, with all 500 claim-bootstrap draws available. All four unchanged controls reproduce their existing primary predictions **exactly** on source 9/9 and target 6/6, including labels, duplicate-disjoint masks, claim-group IDs, and NLL predictions. No integrity errors or estimator failures are recorded. All eleven predetermined component drops remain included; coefficients were not refit.

Source features are from **Llama3.1-8b-instruct**; target features are from **Llama3.2-1B-Instruct using the original shared tokenizer**. Source-nine paper AUROC for the unchanged original/neutral/diverse/compositional programs is **69.973 / 73.137 / 73.007 / 69.538**, versus reproduced NLL **68.711**. These are screen winners frozen by source PopQA validation, one writer seed per policy.

The table shows **full minus dropped-component** paper AUROC, in percentage points, with marginal paired claim-bootstrap 95% intervals. Positive means the component helps at its original fixed coefficient. Source9 and target6 are different benchmark scopes; their component-effect sizes are not direct cross-model comparisons.

| Policy / removed component | Source9 Δ [95% CI] | Target6 Δ [95% CI] |
| --- | ---: | ---: |
| Original / normalized slope | +5.664 [4.307, 6.724] | +5.811 [4.051, 7.222] |
| Original / inverse tail range | +0.760 [0.205, 1.303] | +1.119 [0.091, 2.166] |
| Neutral / total NLL | +0.134 [−0.104, 0.356] | +0.044 [−0.287, 0.337] |
| Neutral / weighted gap | +0.335 [−0.171, 0.613] | +0.162 [−0.238, 0.533] |
| Neutral / tail-gap variance | +1.362 [1.242, 1.517] | +1.057 [0.880, 1.260] |
| Diverse / total NLL | +5.367 [4.552, 6.213] | +5.756 [4.621, 6.912] |
| Diverse / relative-gap drift | −0.004 [−0.011, 0.004] | −0.006 [−0.016, 0.004] |
| Diverse / tail-gap variance | +4.150 [3.645, 4.538] | +3.466 [2.939, 4.036] |
| Compositional / tail-surprisal variance | +0.011 [−0.244, 0.246] | −0.022 [−0.189, 0.117] |
| Compositional / tail-surprisal mean | −1.106 [−1.455, −0.746] | −2.325 [−2.904, −1.807] |
| Compositional / global correlation | +11.279 [10.656, 11.865] | +12.217 [11.239, 13.040] |

**Supported observations.** Tail-gap variance contributes positively in both gap-based formulas; diverse also relies strongly on its explicit NLL term. Diverse's relative-gap drift has negligible aggregate effect at coefficient 2, while compositional's tail variance adds little detectable macro discrimination. The original and compositional formulas rely substantially on position/log-probability correlation. Compositional's tail mean hurts both reported macros at coefficient 0.3. This motivates a future validation-only coefficient study; it does not authorize presenting the test-improved drop as a newly selected winner.

**Direction matters.** The original slope effect is **+33.042 pp directed** but **+5.664 pp folded** on source9: several slope-dropped rankings reverse direction, and folding rewards their remaining discrimination. On Cities, for example, full versus slope-dropped directed AUROC is **94.291 versus 13.442**; folding the latter yields **86.558**. Neutral's explicit NLL actually reduces directed macro AUROC by **0.445 pp** (CI [−0.685, −0.229]), although its folded effect is small and unresolved. Neutral weighted gap improves directed AUROC by **1.079 pp** ([0.745, 1.382]), but its folded interval includes zero. The paper convention uses test labels to choose orientation separately per dataset; it supports paper comparison, not a claim of uniformly correct hallucination-score direction. Per-dataset effects also vary: a helpful macro component need not help every dataset.

**Mathematical and numerical limits.** These are fixed-final-term ablations, not independent feature interventions. NLL, weighted gap, and gap variance share token log probabilities; removing explicit NLL does not remove probability information. The gap uses a softmax normalized over the stored **top 512**, so it includes a truncated-mass correction and is not the exact full-vocabulary top-1 gap. NLL and weighted gap scale with length, unlike normalized drift and variance; near-zero drift effects do not establish why the term is small without component-scale measurements. Diverse's denominator floor of `1e-6` can amplify small initial gaps; it does not damp them. Original's shared surprise normalization remains after either term is dropped. Tail windows and global correlation cover the cached chat-formatted claim, including template tokens, rather than an independently generated answer. Float16 feature storage, short tail windows, and constant inputs can affect variance/correlation; no numerical failure was observed here. Existing guards are preserved, and cache preflight excludes the short-sequence branches.

On **matched source6 versus target6**, all four full programs lose paper AUROC: original **−1.062**, neutral **−3.111**, diverse **−3.153**, compositional **−3.555 pp**; NLL loses **−3.214 pp**. Each paired interval excludes zero. Thus several component patterns persist, but absolute quality is lower with these target features. This is six-of-nine transfer evidence, not a full-nine target result.

The intervals condition on the frozen programs and these datasets. They do not include writer/search-seed variability, validation-winner selection uncertainty, or uncertainty over a broader population of datasets; eleven component comparisons across models/metrics have no multiplicity correction. All drops were included before their test evaluation, but using the resulting table to choose a simplified detector or tune coefficients would make that subsequent choice test-driven. A new selected detector needs validation-based selection and fresh confirmation. These results explain particular evolved formulas and reproduce the stated baseline convention; they do **not** establish a prompt-policy ranking or a new paper-wide/SOTA result.

Sources: completed remote `screen_component_ablations/ABLATION_REPORT.json` and `.md`; frozen ablation `manifest.json`/`DESIGN.md`; [source-candidate inspection](SCREEN_CANDIDATE_INSPECTION.md). Exact per-dataset numbers, all method scores, bootstrap counts, source hashes, and parity receipts remain in the full ablation report.
