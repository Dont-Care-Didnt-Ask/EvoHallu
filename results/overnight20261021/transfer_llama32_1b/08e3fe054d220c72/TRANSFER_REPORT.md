# Frozen-program transfer to Llama3.2-1B

This is a **six-of-nine dataset transfer control**, not the full paper benchmark. Source and target averages use exactly the same 6 datasets. No target labels are used for program selection or fitting.

Source features: Llama3.1-8b-instruct. Target features: Llama3.2-1B-Instruct caches, using the original shared tokenizer.

Only full-budget source-selected primary identities are transferred. Excluded partial/unverified primary identities: 0; details remain in the protocol.

| Frozen source-selected method | Target datasets | Matched source datasets | Source same-6 paper AUROC | Target same-6 paper AUROC | Target − source |
| --- | ---: | ---: | ---: | ---: | ---: |
| original/SequenceLogProb | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next27b_evo_diverse_2107__primary | 6/6 | 6/6 | 73.811 | 70.605 | -3.207 |
| new/next27b_evo_original_2107__primary | 6/6 | 6/6 | 71.994 | 69.098 | -2.895 |
| new/next_evo_compositional_2107__primary | 5/6 | 5/6 | pending | pending | pending |
| new/next_evo_compositional_4319__primary | 6/6 | 6/6 | 70.236 | 67.771 | -2.465 |
| new/next_evo_compositional_8741__primary | 6/6 | 6/6 | 69.659 | 67.130 | -2.529 |
| new/next_evo_diverse_2107__primary | 6/6 | 6/6 | 72.113 | 69.415 | -2.699 |
| new/next_evo_diverse_4319__primary | 6/6 | 6/6 | 73.715 | 70.352 | -3.363 |
| new/next_evo_diverse_8741__primary | 6/6 | 6/6 | 71.006 | 67.790 | -3.216 |
| new/next_evo_neutral_2107__primary | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next_evo_neutral_4319__primary | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next_evo_neutral_8741__primary | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next_evo_neutral_interface_2107__primary | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next_evo_neutral_rank_uniform_2107__primary | 6/6 | 6/6 | 71.006 | 67.790 | -3.216 |
| new/next_evo_neutral_rank_uniform_4319__primary | 6/6 | 6/6 | 71.582 | 68.418 | -3.164 |
| new/next_evo_neutral_rank_uniform_8741__primary | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next_evo_original_2107__primary | 6/6 | 6/6 | 71.583 | 68.417 | -3.166 |
| new/next_evo_original_4319__primary | 6/6 | 6/6 | 72.004 | 69.436 | -2.568 |
| new/next_evo_original_8741__primary | 6/6 | 6/6 | 70.397 | 68.640 | -1.758 |
| new/next_fixed_neutral_2107__primary | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next_fixed_neutral_4319__primary | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next_fixed_neutral_8741__primary | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next_prompt_compositional_2107__primary | 6/6 | 6/6 | 70.494 | 66.939 | -3.555 |
| new/next_prompt_diverse_2107__primary | 6/6 | 6/6 | 74.502 | 71.349 | -3.153 |
| new/next_prompt_neutral_2107__primary | 6/6 | 6/6 | 74.267 | 71.156 | -3.111 |
| new/next_prompt_original_2107__primary | 6/6 | 6/6 | 70.366 | 69.304 | -1.062 |
| new/nextglm_evo_diverse_2107__primary | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/nextglm_evo_original_2107__primary | 6/6 | 6/6 | 71.794 | 69.275 | -2.519 |
| supervised/logistic_documented_claim_group | 6/6 | 6/6 | 74.529 | 70.618 | -3.912 |
| supervised/qda_documented_claim_group | 6/6 | 6/6 | 73.243 | 65.694 | -7.549 |

Prespecified datasets:

- `AVeriTeC__dev_filtered_10shot`
- `UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot`
- `WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot`
- `geometry__cities`
- `geometry__common_claim_true_false`
- `geometry__companies_true_false`

Estimator failures on target datasets:

- {'dataset': 'UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot', 'method': 'new/next_evo_compositional_2107__primary', 'error': 'ValueError: nonfinite scores or wrong output shape'}
