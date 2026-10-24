# Frozen-program transfer to Llama3.2-1B

This is a **six-of-nine dataset transfer control**, not the full paper benchmark. Source and target averages use exactly the same six datasets. No target labels are used for program selection or fitting.

Source features: Llama3.1-8b-instruct. Target features: preexisting Llama3.2-1B-Instruct caches, using the original shared tokenizer.

| Frozen source-selected method | Target datasets | Matched source datasets | Source same-six paper AUROC | Target same-six paper AUROC | Target − source |
| --- | ---: | ---: | ---: | ---: | ---: |
| original/SequenceLogProb | 0/6 | 0/6 | pending | pending | pending |
| supervised/logistic_documented_claim_group | 0/6 | 0/6 | pending | pending | pending |
| supervised/qda_documented_claim_group | 0/6 | 0/6 | pending | pending | pending |

Prespecified datasets:

- `AVeriTeC__dev_filtered_10shot`
- `UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot`
- `WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot`
- `geometry__cities`
- `geometry__common_claim_true_false`
- `geometry__companies_true_false`
