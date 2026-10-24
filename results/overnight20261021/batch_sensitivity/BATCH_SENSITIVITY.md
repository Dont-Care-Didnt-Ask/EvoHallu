# Frozen compositional8741 batch sensitivity

Status: **complete_fixed_five_case_diagnostic**. All five conditions were fixed before scoring; no condition replaces the original512 primary. Source SHA256 `a96fdf5af49a1a6f0f6426d79e2ef60b596be1d7ffd38a672afb6338523cc884`. No model/GPU calls, new generation or refitting.

Original512 must reproduce all nine saved primary predictions, labels, NLL, claim-group IDs and duplicate-disjoint masks byte-for-byte before variant scoring. The original fixed512 protocol remains authoritative.

| Condition | Datasets | Directed macro | Folded macro |
| --- | ---: | ---: | ---: |
| original_128 |9/9|65.9865|68.5457|
| original_512 |9/9|66.5122|68.9655|
| original_2048 |9/9|66.6387|69.1210|
| original_full |9/9|68.3729|70.8474|
| shuffle451_512 |9/9|68.3273|70.7902|

| Dataset / condition | Directed | Folded | Rank rho vs512 |
| --- | ---: | ---: | ---: |
|AVeriTeC__dev_filtered_10shot /original_128|62.7627|62.7627|0.9780919026925723|
|AVeriTeC__dev_filtered_10shot /original_512|63.8431|63.8431|1.0|
|AVeriTeC__dev_filtered_10shot /original_2048|63.8431|63.8431|1.0|
|AVeriTeC__dev_filtered_10shot /original_full|63.8431|63.8431|1.0|
|AVeriTeC__dev_filtered_10shot /shuffle451_512|63.8431|63.8431|1.0|
|UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot /original_128|38.4834|61.5166|0.9852798588257924|
|UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot /original_512|38.9601|61.0399|1.0|
|UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot /original_2048|38.8295|61.1705|0.9961157272198653|
|UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot /original_full|38.8648|61.1352|0.9955598669655237|
|UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot /shuffle451_512|38.9171|61.0829|0.9924575878551106|
|WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot /original_128|65.2453|65.2453|0.9776998885175995|
|WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot /original_512|65.6806|65.6806|1.0|
|WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot /original_2048|65.7676|65.7676|0.9951576107178964|
|WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot /original_full|65.8819|65.8819|0.993621509469387|
|WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot /shuffle451_512|65.8497|65.8497|0.9919277240564597|
|geometry__cities /original_128|91.1845|91.1845|0.9957470724350129|
|geometry__cities /original_512|91.1223|91.1223|1.0|
|geometry__cities /original_2048|91.0640|91.0640|0.9987802057570891|
|geometry__cities /original_full|91.0640|91.0640|0.9987802057570891|
|geometry__cities /shuffle451_512|90.9984|90.9984|0.9975174894821139|
|geometry__common_claim_true_false /original_128|50.0646|50.0646|0.9832352346760916|
|geometry__common_claim_true_false /original_512|51.8898|51.8898|1.0|
|geometry__common_claim_true_false /original_2048|53.2041|53.2041|0.9941021247352418|
|geometry__common_claim_true_false /original_full|68.3567|68.3567|0.9441695199064313|
|geometry__common_claim_true_false /shuffle451_512|68.3557|68.3557|0.941594535883072|
|geometry__companies_true_false /original_128|84.0174|84.0174|0.9878272691976931|
|geometry__companies_true_false /original_512|84.3801|84.3801|0.9999999999999999|
|geometry__companies_true_false /original_2048|84.4983|84.4983|0.9969584995869955|
|geometry__companies_true_false /original_full|84.4983|84.4983|0.9969584995869955|
|geometry__companies_true_false /shuffle451_512|84.3722|84.3722|0.9953326581766088|
|geometry__counterfact_true_false /original_128|71.2318|71.2318|0.9910881607640528|
|geometry__counterfact_true_false /original_512|71.0484|71.0484|1.0|
|geometry__counterfact_true_false /original_2048|71.0061|71.0061|0.9983646611944145|
|geometry__counterfact_true_false /original_full|71.0006|71.0006|0.9977288872234962|
|geometry__counterfact_true_false /shuffle451_512|70.9576|70.9576|0.9959171189175618|
|popqa__popqa_test_filtered_10shot /original_128|76.5650|76.5650|0.9928262446000814|
|popqa__popqa_test_filtered_10shot /original_512|76.7626|76.7626|1.0|
|popqa__popqa_test_filtered_10shot /original_2048|76.7805|76.7805|0.9981750479713801|
|popqa__popqa_test_filtered_10shot /original_full|76.8435|76.8435|0.9977451590301964|
|popqa__popqa_test_filtered_10shot /shuffle451_512|76.7190|76.7190|0.9931644782594417|
|x-fact__test_filtered_10shot /original_128|54.3233|54.3233|0.9850473220085709|
|x-fact__test_filtered_10shot /original_512|54.9227|54.9227|1.0|
|x-fact__test_filtered_10shot /original_2048|54.7550|54.7550|0.9977045946650689|
|x-fact__test_filtered_10shot /original_full|55.0032|55.0032|0.9939250747673424|
|x-fact__test_filtered_10shot /shuffle451_512|54.9330|54.9330|0.9913226677627449|

## Original-order label composition

|Dataset|512chunk prevalence min|max|last chunk rows|
|---|---:|---:|---:|
|AVeriTeC__dev_filtered_10shot|0.7143|0.7143|427|
|UHEAD_bio_gemma_annotated_by_mistral__test_filtered_10shot|0.6667|0.7891|9|
|WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot|0.0918|0.6154|13|
|geometry__cities|0.5000|0.5000|472|
|geometry__common_claim_true_false|0.0000|1.0000|354|
|geometry__companies_true_false|0.4824|0.5227|176|
|geometry__counterfact_true_false|0.5000|0.5000|220|
|popqa__popqa_test_filtered_10shot|0.7207|0.7988|318|
|x-fact__test_filtered_10shot|0.5281|0.6699|267|

The class robust-normalizes each of four feature vectors using the current batch median/MAD; for one row every MAD is0 and the score is0 analytically. This is batch-sensitive unsupervised normalization, not target-label fitting. Original order and the seed451 random permutation were specified without inspecting labels. Their label-prevalence ranges are descriptive, not a new sorting rule.

Differences across these fixed cases measure protocol sensitivity of one already-selected program. Do not select the highest-AUROC case, promote another chunk size, or reinterpret these numbers as a new independent detector result. No intervals are asserted. All nine datasets are required for a macro; partial runs have none.
