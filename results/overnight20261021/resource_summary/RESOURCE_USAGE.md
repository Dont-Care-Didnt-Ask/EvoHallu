# Overnight resource usage

Saved telemetry through 2026-10-22T00:44:41.260340+00:00. Elapsed compute estimate: **$23.16** over 6.47 hours at $3.58/hour.

| GPU | Samples | Mean sampled utilization | Samples ≥ load threshold | Mean utilization in loaded samples | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 378 | 44.3% | 54.5% | 81.2% | 70.94 GiB |
| 1 | 378 | 38.6% | 47.1% | 81.9% | 70.85 GiB |

Load threshold: 10%. Instantaneous nvidia-smi samples approximately once a minute; means and loaded fractions describe samples, not continuous GPU busy time. Allocated memory includes reserved KV cache.

Persisted requests: **1500**; prompt tokens **1,889,962**; completion tokens **2,204,266**; total **4,094,228**. Requests without reported usage: 2.

Counts each persisted candidates.jsonl request once across analysis-manifest jobs. Responses still awaiting evaluation/persistence and unlisted jobs are excluded. Request time sums include overlapping requests and are not wall time.

Elapsed compute estimate at the recorded total pod rate; excludes storage/network and does not certify actual billing or pod stop time.
