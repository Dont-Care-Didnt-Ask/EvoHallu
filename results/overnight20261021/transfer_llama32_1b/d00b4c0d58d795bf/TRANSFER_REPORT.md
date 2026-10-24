# Frozen-program transfer to Llama3.2-1B

This is a **nine-of-nine dataset feature-model transfer control**. The target feature model differs from the paper's source feature model. Source and target averages use exactly the same 9 datasets. No target labels are used for program selection or fitting.

Source features: Llama3.1-8b-instruct. Target features: Llama3.2-1B-Instruct caches, using the original shared tokenizer.

Only full-budget source-selected primary identities are transferred. Excluded partial/unverified primary identities: 0; details remain in the protocol.

| Frozen source-selected method | Target datasets | Matched source datasets | Source same-9 paper AUROC | Target same-9 paper AUROC | Target − source |
| --- | ---: | ---: | ---: | ---: | ---: |
| original/SequenceLogProb | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/next27b_evo_diverse_2107__primary | 9/9 | 9/9 | 72.306 | 67.780 | -4.526 |
| new/next27b_evo_original_2107__primary | 9/9 | 9/9 | 71.155 | 67.471 | -3.684 |
| new/next_evo_compositional_2107__primary | 8/9 | 8/9 | pending | pending | pending |
| new/next_evo_compositional_4319__primary | 9/9 | 9/9 | 70.363 | 66.866 | -3.497 |
| new/next_evo_compositional_8741__primary | 9/9 | 9/9 | 68.965 | 64.995 | -3.971 |
| new/next_evo_diverse_2107__primary | 9/9 | 9/9 | 70.345 | 66.330 | -4.016 |
| new/next_evo_diverse_4319__primary | 9/9 | 9/9 | 72.008 | 67.231 | -4.777 |
| new/next_evo_diverse_8741__primary | 9/9 | 9/9 | 68.712 | 64.834 | -3.878 |
| new/next_evo_neutral_2107__primary | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/next_evo_neutral_4319__primary | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/next_evo_neutral_8741__primary | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/next_evo_neutral_interface_2107__primary | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/next_evo_neutral_rank_uniform_2107__primary | 9/9 | 9/9 | 68.712 | 64.834 | -3.878 |
| new/next_evo_neutral_rank_uniform_4319__primary | 9/9 | 9/9 | 69.342 | 65.429 | -3.914 |
| new/next_evo_neutral_rank_uniform_8741__primary | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/next_evo_original_2107__primary | 9/9 | 9/9 | 69.344 | 65.428 | -3.916 |
| new/next_evo_original_4319__primary | 9/9 | 9/9 | 71.167 | 67.696 | -3.470 |
| new/next_evo_original_8741__primary | 9/9 | 9/9 | 69.320 | 66.648 | -2.671 |
| new/next_fixed_neutral_2107__primary | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/next_fixed_neutral_4319__primary | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/next_fixed_neutral_8741__primary | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/next_prompt_compositional_2107__primary | 9/9 | 9/9 | 69.538 | 65.201 | -4.337 |
| new/next_prompt_diverse_2107__primary | 9/9 | 9/9 | 73.007 | 68.847 | -4.160 |
| new/next_prompt_neutral_2107__primary | 9/9 | 9/9 | 73.137 | 68.415 | -4.722 |
| new/next_prompt_original_2107__primary | 9/9 | 9/9 | 69.973 | 67.684 | -2.289 |
| new/nextglm_evo_diverse_2107__primary | 9/9 | 9/9 | 68.711 | 64.835 | -3.877 |
| new/nextglm_evo_original_2107__primary | 9/9 | 9/9 | 70.671 | 67.219 | -3.452 |
| supervised/logistic_documented_claim_group | 9/9 | 9/9 | 73.019 | 68.939 | -4.080 |
| supervised/qda_documented_claim_group | 9/9 | 9/9 | 72.166 | 64.386 | -7.780 |

Prespecified datasets:

- `AVeriTeC__dev_filtered_10shot`
- `UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot`
- `WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot`
- `geometry__cities`
- `geometry__common_claim_true_false`
- `geometry__companies_true_false`
- `geometry__counterfact_true_false`
- `popqa__popqa_test_filtered_10shot`
- `x-fact__test_filtered_10shot`

Estimator failures on target datasets:

- {'dataset': 'UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot', 'method': 'new/next_evo_compositional_2107__primary', 'error': 'ValueError: nonfinite scores or wrong output shape'}
