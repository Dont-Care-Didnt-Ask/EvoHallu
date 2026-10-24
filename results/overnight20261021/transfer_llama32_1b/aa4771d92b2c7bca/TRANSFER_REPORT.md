# Frozen-program transfer to Llama3.2-1B

This is a **six-of-nine dataset transfer control**, not the full paper benchmark. Source and target averages use exactly the same 6 datasets. No target labels are used for program selection or fitting.

Source features: Llama3.1-8b-instruct. Target features: Llama3.2-1B-Instruct caches, using the original shared tokenizer.

Only full-budget source-selected primary identities are transferred. Excluded partial/unverified primary identities: 0; details remain in the protocol.

| Frozen source-selected method | Target datasets | Matched source datasets | Source same-6 paper AUROC | Target same-6 paper AUROC | Target − source |
| --- | ---: | ---: | ---: | ---: | ---: |
| original/SequenceLogProb | 6/6 | 6/6 | 71.006 | 67.791 | -3.214 |
| new/next_prompt_compositional_2107__primary | 6/6 | 6/6 | 70.494 | 66.939 | -3.555 |
| new/next_prompt_diverse_2107__primary | 6/6 | 6/6 | 74.502 | 71.349 | -3.153 |
| new/next_prompt_neutral_2107__primary | 6/6 | 6/6 | 74.267 | 71.156 | -3.111 |
| new/next_prompt_original_2107__primary | 6/6 | 6/6 | 70.366 | 69.304 | -1.062 |
| supervised/logistic_documented_claim_group | 6/6 | 6/6 | 74.529 | 70.618 | -3.912 |
| supervised/qda_documented_claim_group | 6/6 | 6/6 | 73.243 | 65.694 | -7.549 |

Prespecified datasets:

- `AVeriTeC__dev_filtered_10shot`
- `UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot`
- `WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot`
- `geometry__cities`
- `geometry__common_claim_true_false`
- `geometry__companies_true_false`
