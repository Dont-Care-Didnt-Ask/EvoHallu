# Batch-composition sensitivity: interpretation

**Completed CPU diagnostic, 21 October 2026, 22:21 UTC.** The same already-frozen compositional8741 program was scored under all five prospectively fixed conditions. All **nine original 512** score arrays, labels, canonical NLL, group IDs and duplicate-disjoint masks reproduce the primary **byte-for-byte**. All45 dataset-condition NLL checks pass. No model/GPU call, new fit, candidate repair or primary replacement occurred.

|Condition|Directed macro|Folded macro|Folded difference vs original 512|
|---|---:|---:|---:|
|original_128|65.9865|68.5457|-0.4198|
|original_512|66.5122|68.9655|+0.0000|
|original_2048|66.6387|69.1210|+0.1555|
|original_full|68.3729|70.8474|+1.8819|
|shuffle451_512|68.3273|70.7902|+1.8247|

**The issue is row membership within each batch, not a train/test chunk-size mismatch.** The original training, validation and benchmark paths all used 512-row chunks. The program computes four feature vectors, then normalizes each by the median/MAD across the current call. Its effective feature scales therefore depend on the other samples sharing that chunk. For a singleton batch every MAD is zero and its score is zero; this follows algebraically and passed the focused fixture.

**One label-ordered dataset dominates the effect.** CommonClaims has original 512-chunk hallucination prevalences `[0,0,0,0,0.654296875,1,1,1,1]`; its last chunk contains 354 rows. Its AUROC is 50.0646 at 128,51.8898 at 512,53.2041 at 2048,68.3567 with a full dataset batch, and68.3557 with the fixed seed451 permutation at 512. Rank correlations to the original 512 score are 0.94417(full) and0.94159(shuffle). CommonClaims accounts for **+16.4669/9=+1.8297 macro points**, approximately 97% of the full-batch macro change. For full/shuffle, the other eight individual folded changes are each within 0.21 pp.

This is consistent with per-chunk robust centering/scaling changing the class-separating signal when batches have different class mixtures. The normalization itself never reads labels; prevalence is a descriptive audit, and the shuffle was fixed without labels. This one diagnostic does not prove that label composition is the sole mechanism.

**Keep original 512 as the primary result.** Full and shuffled cases are sensitivity measurements, not new discovered detectors or test-selected improvements. No alternate case is promoted. In particular, the favorable full-batch macro does not justify changing evaluation after seeing test labels. No confidence intervals or general policy-level advantage are asserted. The singleton case is an analytic/fixture observation, not an additional benchmark condition.

The copied source remains byte-identical: SHA256 `a96fdf5af49a1a6f0f6426d79e2ef60b596be1d7ffd38a672afb6338523cc884`. Cases 128/2048/full/shuffle use the same repository feature builder with only the declared sequence-logprob dependency retained; the unchanged estimator code and deep-copy scoring wrapper are used. The exact original 512 reference uses the original full logits packing. Full-vs-gated fixture outputs match exactly. Source raw-cache, CSV, training CSV, primary prediction and helper hashes are pinned in [PLAN](PLAN.json).

Runtime: **178.92seconds wall**, at most 4 single-thread CPU workers, CUDA hidden, under the 30-minute cap. [All per-dataset results](BATCH_SENSITIVITY.md), [JSON](BATCH_SENSITIVITY.json), [CSV](BATCH_SENSITIVITY.csv), and [execution receipt](execution.json) are backed up locally. Seven focused fixtures passed; an independent saved-array recomputation reproduced all five reported macros exactly. No raw caches were copied for this diagnostic backup.
