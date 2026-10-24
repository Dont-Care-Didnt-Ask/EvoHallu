# EvoHallu overnight experiments

Completed 27/27 jobs; 1500/1500 requests.

Labels are `is_hallu`. Paper-compatible reporting folds AUROC within each dataset before averaging. Selection uses directed PopQA validation only; every seed is retained.

## Frozen methods

| Method | Interpretation | Datasets | Directed AUROC | Paper-compatible AUROC |
| --- | --- | ---: | ---: | ---: |
| new/next27b_evo_diverse_2107__primary | full_budget_primary | 9/9 | 70.29734295356162 | 72.30606535101663 |
| new/next27b_evo_diverse_2107__trainbest | secondary_selection_diagnostic | 9/9 | 70.29734295356162 | 72.30606535101663 |
| new/next27b_evo_original_2107__primary | full_budget_primary | 9/9 | 69.0600901022476 | 71.15468516741143 |
| new/next27b_evo_original_2107__trainbest | secondary_selection_diagnostic | 9/9 | 68.8064560822074 | 70.96065551199048 |
| new/next_evo_compositional_2107__primary | full_budget_primary | 9/9 | 69.07518974001833 | 70.6768969055287 |
| new/next_evo_compositional_2107__trainbest | secondary_selection_diagnostic | 9/9 | 69.07518974001833 | 70.6768969055287 |
| new/next_evo_compositional_4319__primary | full_budget_primary | 9/9 | 69.30398579820877 | 70.36332543450943 |
| new/next_evo_compositional_4319__trainbest | secondary_selection_diagnostic | 9/9 | 69.30398579820877 | 70.36332543450943 |
| new/next_evo_compositional_8741__primary | full_budget_primary | 9/9 | 66.51218405349672 | 68.96549703547143 |
| new/next_evo_compositional_8741__trainbest | secondary_selection_diagnostic | 9/9 | 66.51218405349672 | 68.96549703547143 |
| new/next_evo_diverse_2107__primary | full_budget_primary | 9/9 | 67.7442605545414 | 70.34543211081142 |
| new/next_evo_diverse_2107__trainbest | secondary_selection_diagnostic | 9/9 | 67.7442605545414 | 70.34543211081142 |
| new/next_evo_diverse_4319__primary | full_budget_primary | 9/9 | 69.85784898904942 | 72.00788072038985 |
| new/next_evo_diverse_4319__trainbest | secondary_selection_diagnostic | 9/9 | 69.85784898904942 | 72.00788072038985 |
| new/next_evo_diverse_8741__primary | full_budget_primary | 9/9 | 65.6431497469665 | 68.71167185132946 |
| new/next_evo_diverse_8741__trainbest | secondary_selection_diagnostic | 9/9 | 68.9258901033022 | 71.84478100838287 |
| new/next_evo_neutral_2107__primary | full_budget_primary | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_2107__trainbest | secondary_selection_diagnostic | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_4319__primary | full_budget_primary | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_4319__trainbest | secondary_selection_diagnostic | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_8741__primary | full_budget_primary | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_8741__trainbest | secondary_selection_diagnostic | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_interface_2107__primary | full_budget_primary | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_interface_2107__trainbest | secondary_selection_diagnostic | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_rank_uniform_2107__primary | full_budget_primary | 9/9 | 65.6431497469665 | 68.71167185132946 |
| new/next_evo_neutral_rank_uniform_2107__trainbest | secondary_selection_diagnostic | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_rank_uniform_4319__primary | full_budget_primary | 9/9 | 66.30483673749029 | 69.3423717333023 |
| new/next_evo_neutral_rank_uniform_4319__trainbest | secondary_selection_diagnostic | 9/9 | 66.30483673749029 | 69.3423717333023 |
| new/next_evo_neutral_rank_uniform_8741__primary | full_budget_primary | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_neutral_rank_uniform_8741__trainbest | secondary_selection_diagnostic | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_evo_original_2107__primary | full_budget_primary | 9/9 | 66.30800284628233 | 69.34362990222286 |
| new/next_evo_original_2107__trainbest | secondary_selection_diagnostic | 9/9 | 67.92407904467396 | 68.57560668206253 |
| new/next_evo_original_4319__primary | full_budget_primary | 9/9 | 69.09450133739455 | 71.16660533896743 |
| new/next_evo_original_4319__trainbest | secondary_selection_diagnostic | 9/9 | 69.09450133739455 | 71.16660533896743 |
| new/next_evo_original_8741__primary | full_budget_primary | 9/9 | 68.06699767748415 | 69.31970688229632 |
| new/next_evo_original_8741__trainbest | secondary_selection_diagnostic | 9/9 | 68.06699767748415 | 69.31970688229632 |
| new/next_fixed_neutral_2107__primary | full_budget_primary | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_fixed_neutral_2107__trainbest | secondary_selection_diagnostic | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_fixed_neutral_4319__primary | full_budget_primary | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_fixed_neutral_4319__trainbest | secondary_selection_diagnostic | 9/9 | 66.30039041189929 | 66.57525553678816 |
| new/next_fixed_neutral_8741__primary | full_budget_primary | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_fixed_neutral_8741__trainbest | secondary_selection_diagnostic | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/next_prompt_compositional_2107__primary | full_budget_primary | 9/9 | 67.4654161890063 | 69.53768969833672 |
| new/next_prompt_diverse_2107__primary | full_budget_primary | 9/9 | 70.13072898708211 | 73.00718514438294 |
| new/next_prompt_neutral_2107__primary | full_budget_primary | 9/9 | 70.84821025303067 | 73.136560859309 |
| new/next_prompt_original_2107__primary | full_budget_primary | 9/9 | 68.7713368114169 | 69.97286532835223 |
| new/nextglm_evo_diverse_2107__primary | full_budget_primary | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/nextglm_evo_diverse_2107__trainbest | secondary_selection_diagnostic | 9/9 | 65.64388231516448 | 68.71117305407847 |
| new/nextglm_evo_original_2107__primary | full_budget_primary | 9/9 | 68.53308417221132 | 70.67107183160111 |
| new/nextglm_evo_original_2107__trainbest | secondary_selection_diagnostic | 9/9 | 68.53308417221132 | 70.67107183160111 |
| original/SequenceLogProb | fixed_fitted_or_reference_method | 9/9 | 65.64388231516448 | 68.71117305407847 |
| supervised/logistic_documented_claim_group | fixed_fitted_or_reference_method | 9/9 | 70.6154011889269 | 73.01940542698907 |
| supervised/qda_documented_claim_group | fixed_fitted_or_reference_method | 9/9 | 69.68983034160945 | 72.16571384631908 |

## Baseline reproducibility

Exact historical comparison: **passed**, 9/9 datasets checked. Paper-compatible NLL AUROC: 68.71117305407847. Labels remain `is_hallu`.

Byte-identical historical source CSVs; exact rowwise labels, NLL predictions, claim-group IDs and train-disjoint masks

## All-seed primary results

Only full-budget primary identities enter these groups. Standard deviations below describe variation across the completed seeds; they are not confidence intervals.

| Phase / policy | Writer | Evaluated seeds | Paper AUROC mean | Seed SD |
| --- | --- | --- | ---: | ---: |
| evolution_screen / compositional | Qwen/Qwen3.5-9B | [2107, 4319, 8741] | 70.00190645850319 | 0.9111477026097683 |
| evolution_screen / diverse | Qwen/Qwen3.5-9B | [2107, 4319, 8741] | 70.35499489417691 | 1.648125241672298 |
| evolution_screen / neutral | Qwen/Qwen3.5-9B | [2107, 4319, 8741] | 68.71117305407847 | 0.0 |
| evolution_screen / original | Qwen/Qwen3.5-9B | [2107, 4319, 8741] | 69.9433140411622 | 1.0594688656044124 |
| feedback_control / neutral | Qwen/Qwen3.5-9B | [2107, 4319, 8741] | 68.71117305407847 | 0.0 |
| interface_control / neutral | Qwen/Qwen3.5-9B | [2107] | 68.71117305407847 | None |
| prompt_screen / compositional | Qwen/Qwen3.5-9B | [2107] | 69.53768969833672 | None |
| prompt_screen / diverse | Qwen/Qwen3.5-9B | [2107] | 73.00718514438294 | None |
| prompt_screen / neutral | Qwen/Qwen3.5-9B | [2107] | 73.136560859309 | None |
| prompt_screen / original | Qwen/Qwen3.5-9B | [2107] | 69.97286532835223 | None |
| sampler_control / neutral | Qwen/Qwen3.5-9B | [2107, 4319, 8741] | 68.92173887957007 | 0.36427882237217774 |
| writer_confirmation / diverse | Qwen/Qwen3.8-27B | [2107] | 72.30606535101663 | None |
| writer_confirmation / original | Qwen/Qwen3.8-27B | [2107] | 71.15468516741143 | None |
| writer_generalization / diverse | zai-org/GLM-4.7-Flash | [2107] | 68.71117305407847 | None |
| writer_generalization / original | zai-org/GLM-4.7-Flash | [2107] | 70.67107183160111 | None |

## Matched comparisons

Only completed nine-dataset primary methods with full planned budgets are paired. Writers, seeds and request counts must match. Positive deltas favor the named arm over its reference.

| Comparison | Arm | Reference | Matched seeds | Paper AUROC delta, pp |
| --- | --- | --- | --- | ---: |
| fixed_context_prompt_reference | prompt_screen / compositional | prompt_screen / original | [2107] | -0.4351756300155074 |
| fixed_context_prompt_reference | prompt_screen / diverse | prompt_screen / original | [2107] | 3.0343198160307168 |
| fixed_context_prompt_reference | prompt_screen / neutral | prompt_screen / original | [2107] | 3.1636955309567725 |
| fixed_examples_vs_neutral_evolution | feedback_control / neutral | evolution_screen / neutral | [2107, 4319, 8741] | 0.0 |
| interface_metric_contract_vs_neutral_evolution | interface_control / neutral | evolution_screen / neutral | [2107] | 0.0 |
| original_evolution_reference | evolution_screen / compositional | evolution_screen / original | [2107, 4319, 8741] | 0.05859241734098456 |
| original_evolution_reference | evolution_screen / diverse | evolution_screen / original | [2107, 4319, 8741] | 0.41168085301470586 |
| original_evolution_reference | evolution_screen / neutral | evolution_screen / original | [2107, 4319, 8741] | -1.232140987083729 |
| original_evolution_reference | feedback_control / neutral | evolution_screen / original | [2107, 4319, 8741] | -1.232140987083729 |
| original_evolution_reference | interface_control / neutral | evolution_screen / original | [2107] | -0.6324568481443862 |
| original_evolution_reference | sampler_control / neutral | evolution_screen / original | [2107, 4319, 8741] | -1.021575161592125 |
| rank_sampling_vs_neutral_evolution | sampler_control / neutral | evolution_screen / neutral | [2107, 4319, 8741] | 0.21056582549160416 |
| writer_confirmation_prompt_reference | writer_confirmation / diverse | writer_confirmation / original | [2107] | 1.1513801836051982 |
| writer_generalization_prompt_reference | writer_generalization / diverse | writer_generalization / original | [2107] | -1.959898777522639 |

## Training-probe diagnostics

The fixed 512-row training probe is separate from validation. Rank similarity ≥0.995 is the prespecified near-duplicate diagnostic.

| Run | Status | Requests | Valid | Distinct rankings | Near-identical sibling pairs / comparable | Validation AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| next_prompt_original_2107 | finished | 30 | 19 | 18 | 0/6 | 77.7002 |
| next_prompt_neutral_2107 | finished | 30 | 19 | 17 | 0/4 | 80.4055 |
| next_prompt_diverse_2107 | finished | 30 | 23 | 22 | 0/7 | 79.513 |
| next_prompt_compositional_2107 | finished | 30 | 11 | 11 | 0/3 | 80.2857 |
| next_evo_original_2107 | finished | 60 | 30 | 29 | 1/6 | 76.9672 |
| next_evo_neutral_2107 | finished | 60 | 29 | 21 | 1/8 | 75.7504 |
| next_evo_diverse_2107 | finished | 60 | 41 | 26 | 0/9 | 76.8771 |
| next_evo_original_4319 | finished | 60 | 36 | 32 | 1/9 | 79.2061 |
| next_evo_neutral_4319 | finished | 60 | 36 | 17 | 3/10 | 75.7504 |
| next_evo_diverse_4319 | finished | 60 | 25 | 18 | 0/4 | 79.5207 |
| next_evo_original_8741 | finished | 60 | 35 | 35 | 1/9 | 78.454 |
| next_evo_neutral_8741 | finished | 60 | 33 | 17 | 2/8 | 75.7504 |
| next_evo_diverse_8741 | finished | 60 | 37 | 29 | 0/9 | 75.7622 |
| next_evo_compositional_2107 | finished | 60 | 27 | 26 | 0/8 | 76.8237 |
| next_evo_compositional_4319 | finished | 60 | 29 | 27 | 1/8 | 76.6394 |
| next_evo_compositional_8741 | finished | 60 | 22 | 22 | 2/5 | 78.2742 |
| next_fixed_neutral_2107 | finished | 60 | 27 | 15 | 0/5 | 75.7504 |
| next_fixed_neutral_4319 | finished | 60 | 30 | 16 | 2/6 | 75.7504 |
| next_fixed_neutral_8741 | finished | 60 | 31 | 17 | 2/7 | 75.7504 |
| next_evo_neutral_rank_uniform_2107 | finished | 60 | 35 | 26 | 3/11 | 75.7622 |
| next_evo_neutral_rank_uniform_4319 | finished | 60 | 33 | 17 | 2/7 | 76.9672 |
| next_evo_neutral_rank_uniform_8741 | finished | 60 | 32 | 21 | 0/11 | 75.7504 |
| next_evo_neutral_interface_2107 | finished | 60 | 29 | 19 | 2/7 | 75.7504 |
| next27b_evo_original_2107 | finished | 60 | 52 | 49 | 6/25 | 79.5366 |
| next27b_evo_diverse_2107 | finished | 60 | 56 | 42 | 0/25 | 80.1645 |
| nextglm_evo_original_2107 | finished | 60 | 28 | 22 | 0/3 | 78.7667 |
| nextglm_evo_diverse_2107 | finished | 60 | 36 | 20 | 1/6 | 75.7504 |

## Interpretation limits

Paper references are rounded context, not matched significance tests: NLL 68.7, GPT 70.8, Claude 73.4

Intervals condition on each frozen program; they do not include variability of writer training/search

Optional validation shortlist entries are secondary diagnostics; no test-best program or test-best seed is promoted to the primary result.
