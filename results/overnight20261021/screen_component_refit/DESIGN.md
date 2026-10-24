# Exploratory source-trained combination of frozen screen components

This is **one CPU-only fitted control**, prepared after the screen programs and component-ablation pool were frozen. It asks whether training coefficients for the existing LLM-proposed signals improves over the writers' fixed coefficients and the existing documented-14-feature logistic control. It is not joint supervised evolution, a new writer run, feature selection by OOD quality, or feature-model training.

The source pool contains **all four screen winners** (original, neutral, diverse, compositional) and **all eleven additive components** in the existing ablation manifest. No policy or component is excluded. Each feature is computed exactly as the actual frozen full-program output minus the corresponding frozen drop-program output, including float32 rounding and branch behaviour. This is preferable to reimplementing the writers' formulas with subtly different alignment or short-sequence handling. A twelfth, predefined column, **`float64_total_nll`**, is unconditional total NLL, summed after promoting token log-probabilities to float64. The benchmark's canonical `SequenceLogProb` instead preserves the input float16 sum and returns float32, so these numerical implementations are not identified as the same baseline. The float64 feature preserves NLL information where both full and drop programs take the same short-sequence fallback. The resulting pool intentionally contains correlated/redundant NLL-like columns. There is no rank, variance, significance, or test-quality pruning, and individual fitted coefficients should not be interpreted as independent feature importance.

## Fit and validation

- Existing pinned PopQA training cache: 7,007 rows, SHA-256 `8d34da6f189e8bfe67c0bca3e1101607f5b0887d5f1de41b4f373ef08f2f5ced`.
- Same normalized-claim grouped split as the documented-14 control: seed 441, **5,606 fitting / 1,401 outer-validation rows**, no normalized-claim overlap.
- Three-fold `StratifiedGroupKFold` inside the fitting subset only, shuffle and seed 441.
- `StandardScaler` stays inside every CV pipeline. Logistic regression uses `max_iter=2000` and the same five C values: **0.01, 0.1, 1, 10, 100**. The highest mean inner directed ROC-AUC selects C; ties retain the first predefined value.
- Fit the chosen pipeline on the 5,606 fitting rows only. Outer validation does not select C or refit the model. No test/target labels enter fitting.
- **The feature pool is validation-informed:** its source programs were previously selected using PopQA validation, with a different historical partition. Therefore the new outer-validation AUC is a diagnostic, not an independent validation estimate. Group separation prevents direct normalized-claim overlap in this coefficient fit; it cannot erase the programs' earlier selection history.

All nine benchmark results have previously been viewed during the campaign. Any subsequent source-nine or target-six assessment of this head is explicitly **exploratory**. Freeze weights once on the source 8B-feature training cache, then use exactly those weights for source and target evaluations. There is no target refit, feature selection, orientation tuning, or subsequent coefficient adjustment using benchmark results.

## Frozen inference and integration

`component_pool.json` fixes feature order and byte-identical copies of all fifteen programs. `weights.json` will contain plain numeric scaler/linear coefficients, rather than a mutable pickle dependency. The generated ordinary `HallucinationDetector(Estimator)` wrapper guards the weight file and inference-module hashes; the helper guards the pool manifest and all component-source hashes and keeps relative repository paths. Every component receives a private copy of its declared inputs.

The existing cached evaluator can load the wrapper with no core or evaluator changes. It retains the shared NLL reference and existing normalized-claim duplicate-disjoint diagnostics. Source nine and target six are reported separately; six-dataset means must not be compared against the paper's nine-dataset means. Target coverage can be extended to nine only after the missing caches have been completed and verified. Directed `is_hallu` AUC and the paper-style per-dataset `max(AUC,1-AUC)` convention remain separate summaries.

No active execution manifest, frozen primary registry, pinned core file, or existing fitted head is changed. Fitting is explicit (`--execute`); default invocation only prepares the immutable source pool and evaluation commands. Preparation does not open training caches, held-out feature caches, or prediction files. It never calls an LLM or GPU. If a frozen weight file already exists, the fitter refuses to overwrite/refit it.

## Execution artifacts

The explicit fitting command and subsequent source/target commands are in `commands.json`. `run_source_nine.sh` and `run_target_six.sh` score only after a successful fit has generated `frozen/screen_component_logistic/frozen_estimator.py`. The fitter writes `fit_report.json`, `weights.json`, and `training_features_and_predictions.npz` with exact fit/validation indices and frozen predictions; inference serialization must match sklearn's decision function to numerical precision.

The training feature pass evaluates fifteen small frozen programs per chunk, so it can be slower than the existing fourteen hand-coded feature statistics. Run on idle CPU capacity and preserve the original experiment deadlines. It requires no additional GPU allocation. This preparation has not performed real fitting or benchmark scoring.

Two synthetic tests passed in **11.74 seconds**, including the existing cached evaluator's actual frozen-estimator loader. They verify all-column inclusion and exact differences, preserved short-sequence branches, fit-only CV/scaling despite changing outer-validation labels/features, frozen numeric prediction parity, chunk invariance, and source/artifact/module guards. `preparation_validation.json` records these checks and hashes. These are preparation tests, not detector-quality results.
