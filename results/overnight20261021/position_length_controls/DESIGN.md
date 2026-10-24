# Frozen quadratic-position and length diagnostic

Prepared 2026-10-21 UTC after the source-nine 27B results were already viewed. This is a separate exploratory diagnostic, not a new primary search arm or a benchmark-independent confirmation. No new program generation, training, feature extraction, top-k pack, GPU call or model download is required.

The reference is the actual validation-selected `next27b_evo_diverse_2107__primary` program, SHA256 `6a399c68edd3d47e2ad09a611d44204f378514103329a4205422f08a6cb336a3`. It sums float32 observed-token NLL multiplied by the quadratic ramp `(arange(n,float32)/(n-1))**2`. For n=0 it returns zero; for n=1 it uses weight one. The reference is executed from its unchanged frozen file, rather than reimplemented as an alternative candidate.

The four fixed controls use exactly the same complete cached LP span:

| Name | Definition |
| --- | --- |
| `lp_total_length` | Number of stored observed-token logprobs; no claim crop. |
| `float32_nll_sum` | Sum of `-asarray(lp,float32)` with float32 reduction. |
| `float32_nll_mean` | Mean of `-asarray(lp,float32)` with float32 reduction. |
| `float32_quadratic_mean` | The same float32 quadratic weighted sum as the reference, divided **after summation** by the float32 sum of the same weights. |

Every empty sequence yields zero for every control. All outputs follow the reference's final float32 storage, then the existing evaluator's float64 promotion. The historical canonical NLL is retained separately: it sums the stored float16 logprobs before float32 output, so it is not silently renamed float32 NLL.

Uniform sum minus uniform mean probes accumulation; quadratic sum minus quadratic mean probes weight mass; quadratic mean minus uniform mean probes positional emphasis. All three contrasts, all four controls and all negative results remain in the output. AUROC is nonlinear, so these contrasts are not a causal additive decomposition. No highest-scoring control replaces the frozen primary.

Before any control is scored, a first CPU phase must reproduce **all nine** saved frozen27B predictions and canonical NLL arrays byte-for-byte. CSV/cache ordered claims and labels, normalized-claim group IDs, and duplicate-disjoint masks must match saved primary arrays exactly. These checks use the original 512-row chunks and existing LP-only dependency gate, avoiding top-k tensor packing. The source, helper/support code, dataset CSVs, training CSV, expected raw cache SHA256/size, original primary NPZs and historical context are pinned in an immutable `PLAN.json` before workers launch. Each raw cache is checked against its archived historical digest before scoring. Four CPU subprocesses maximum; CUDA hidden and numerical library threads limited to one. A shared 30-minute limit and hard stop at **2026-10-22 02:00 UTC** bound both phases and reporting. Cancellation markers and SIGTERM terminate owned process groups. Default invocation only prepares the plan; `--execute` is required.

The second CPU phase computes the four controls and 500 paired normalized-claim bootstrap draws. Every method shares the same claim-group multiplicities within a dataset, seeded by `441 + int(SHA256(dataset_name)[:8],16)`. Strict sensitivity uses those same draws restricted to claims absent from the complete PopQA training CSV. Report directed and per-dataset folded AUROC, equal-weight nine-dataset macros, paired gains versus canonical NLL and the frozen quadratic reference, and the three fixed contrasts. Fold each dataset within each bootstrap replicate before averaging; intervals use only draws finite for every prescribed dataset and method. No incomplete-coverage macro is reported. Bootstrap intervals condition on this single selected program, rather than estimating writer/search-seed uncertainty.

Existing historical controls provide context, not new experiments: claim-token length 49.368 directed /54.634 folded; claim-only mean surprisal 66.339/66.937; full-sequence `Perplexity` 64.175/64.175; normalized exponential-weighted NLL 69.797/70.606; accumulated linear-position NLL 69.793/72.000. The historical Perplexity implementation is `-exp(mean(LP))`, mathematically monotonic in mean NLL, with float16 reduction/exponential ties. Claim-only controls have a different span. The documented 14-feature head jointly includes total LP length, NLL sum/mean and normalized exponential NLL; those head results do not isolate standalone controls. Historical point estimates and their source SHA are copied into PLAN from [paper_reporting.json](../../next_experiments/paper_reporting.json).

Prepare and execute on the authorized pod only, from `/workspace/mod1`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/overnight_position_length_controls.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/overnight_position_length_controls.py --execute
```

Outputs stay under this directory: `PLAN.json`, `execution.json`, per-dataset reference/controls receipts, compact score/bootstrap NPZs, and `POSITION_LENGTH_REPORT.json/.md`. The original frozen sources, evaluation predictions, primary registries, search manifests and inference helpers remain unchanged. Preparation and tests do not score actual benchmark rows; root decides whether to launch.
