# Prospective train-only normalization control

One exploratory control follows the completed fixed-case batch-sensitivity diagnostic. **No original program, primary result, parent diagnostic or fitted head is changed.** No GPU, generation, target-model evaluation or coefficient tuning is authorized here.

The immutable C8741 class is loaded using the existing loader. A private NumPy proxy transparently captures its four actual `robust_norm` input arrays. Every feature calculation remains original, including the highest-logprob10% term and short-sequence fallbacks. Eight median calls per prediction are checked; the private namespace is restored after each call.

Fit exactly four medians/MADs on the original **5,605 stratified PopQA training rows**, seed441, from the original pinned7007-row raw cache. Capture features on the original512-row training chunks, concatenate only fit rows, and compute training median/MAD. **Do not compute validation features or use validation/test/target statistics for fitting.** Labels are used only to reproduce the original row split. Keep original coefficients0.4/0.25/0.2/0.15, multiplier1.4826 and MAD<1e-8 zero-vector rule.

Freeze parameters before source-nine scoring. First reconstruct the original512 predictions on all nine datasets using a transparent proxy and require byte-identical primary scores, canonical NLL, labels, group IDs and strict masks. Only after all nine pass, score the separately named frozen-normalization control at chunks1/128/512 and fixed seed451 shuffle512, restoring original positions. Require all four predictions and NLL arrays to be byte-identical within every dataset. No best case is chosen.

All nine source datasets and all four invariance cases are retained regardless of quality. Results are exploratory because this control was designed after earlier test results were viewed. No confidence interval, primary replacement or paper-SOTA claim is planned.

Execution is CPU-only, CUDA hidden, at most four single-thread workers, max30 minutes wall and hard deadline **23:15UTC**. Separate output folder and lock/cancellation guards; failures publish no control-quality claim. Original batch helper and its completed receipts remain unchanged.
