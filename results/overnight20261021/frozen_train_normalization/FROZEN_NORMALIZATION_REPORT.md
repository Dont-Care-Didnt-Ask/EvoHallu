# Train-only frozen-normalization control

**complete_exploratory_train_only_control**. One normalization control derived from the immutable C8741 program; all four original coefficients and feature computations retained. Statistics use5605 original PopQA training rows only; validation/test/target feature distributions and labels do not determine normalization. Exploratory study designed after viewing prior test results.

|Method|Source9 directed|Source9 folded|
|---|---:|---:|
|Canonical NLL|65.6439|68.7112|
|Original C8741,512-row primary|66.5122|68.9655|
|Separate train-normalized control|67.7833|70.5585|

|Dataset|Original512 folded|Frozen train-normalized folded|NLL folded|
|---|---:|---:|---:|
|AVeriTeC__dev_filtered_10shot|63.8431|64.4047|63.7678|
|UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot|61.0399|62.4883|63.8028|
|WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot|65.6806|67.4080|70.6859|
|geometry__cities|91.1223|90.9938|91.9370|
|geometry__common_claim_true_false|51.8898|66.8141|63.2608|
|geometry__companies_true_false|84.3801|83.0949|72.5807|
|geometry__counterfact_true_false|71.0484|70.9574|68.0849|
|popqa__popqa_test_filtered_10shot|76.7626|76.8364|74.0517|
|x-fact__test_filtered_10shot|54.9227|52.0284|50.2290|

All9 transparent-proxy original512 predictions/NLL/labels/groups/masks reproduce the saved primary byte-for-byte before control scoring. All36 chunk/order cases have exactly identical frozen-control scores and canonical NLL across1/128/512 and the fixed seed451 shuffle. Original train/validation/benchmark scoring all used512-row chunks; there was no primary train/test chunk mismatch.

The helper only intercepts the eight median calls in the private copy of the original class. It captures the four actual robust_norm input vectors; it does not rewrite feature math or alter original source bytes. Frozen median/MAD returns preserve the existing1.4826 scale and MAD<1e-8 zero-vector rule. The fitting receipt retains exact split indices and source/helper/cache hashes.

No case replaces the primary, no target evaluation or coefficient fitting occurred, and no confidence interval or paper-SOTA claim is made. The control tests whether frozen source-training normalization yields a batch-independent deployable score; its measured quality is exploratory.
