# Early current-runtime numerical reference

**Passed, 21 October 2026 at 21:49:04 UTC.** This isolated pilot regenerated all **427 historical AVeriTeC rows** with the pinned Llama-3.2-1B weights and shared Llama-3.1 tokenizer under the current pod environment. It did not extract the three missing datasets, fit detectors, or compute quality AUROC.

| Check | Observed result |
| --- | --- |
| Claims, labels, input IDs, masks, shapes and dtypes | Exact across all 427 rows |
| Numeric values | All finite |
| Observed-token logprob maximum absolute difference | **0.00390625**, below the predeclared 0.1 limit |
| Canonical float16-summed NLL Spearman correlation | **0.9999998844**, above the predeclared 0.999 limit |
| Canonical NLL maximum absolute difference | 0.125; no bitwise NLL parity claim |
| Top-512 logits | **All 9,032,704 values exact** |
| Top-512 ranked token IDs | All exact |
| Conditional top-1/top-10 mass and entropy | All exact |
| Six historical target-cache SHA256s | All unchanged after execution |
| Child runtime | **31.66 seconds**, within its 180-second cap |

Only the last compositional 9B source run remained active, at **14/60 requests**, on GPU1/port18001. GPU0/port18000 retained its idle 9B server; two prelaunch samples showed **0% utilization and 23,720 MiB free**. Only the two expected serving processes occupied GPUs. The longest reference input had 85 tokens. Model/tokenizer files, historical download manifests and all six original raw-cache hashes were checked before launch; exact historical tokenizer input IDs were reproduced. The isolated extraction used GPU0 only and returned its free memory to 23,720 MiB. No server was stopped or restarted.

The existing `prepare_remaining_transfer.run_numerical_reference` helper supplied the unchanged numerical guard, child cleanup and source/cache provenance. A separate parent watchdog allowed 210 seconds and monitored writer process changes, source completion and cancellation markers. The pilot finished without a monitor abort. Main transfer-plan/completion/numerical-reference receipts were not written; all pilot outputs are under this separate directory.

Evidence: [pilot receipt](pilot_receipt.json), [full numerical comparison](remaining_transfer_numerical_reference.json), and [one-off launcher](run_pilot.py). Exact executed source SHA256s and pinned checkpoint revisions are in the pilot receipt. The launcher was executed from `/tmp/evohallu_early_numeric_reference.py`; its local archived source is byte-identical.

This is a **gross-drift check on one fixed 427-row reference**, not bitwise feature equivalence or validation of every dataset/precision regime. Float16 accumulation can amplify tiny token differences into a discrete summed-NLL step. The formal future three-cache preparer must run its ordinary numerical guard again before extraction.
