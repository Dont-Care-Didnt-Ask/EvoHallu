# Source-fixed C8741 normalization transfer

Status: **complete_exploratory_nine**.

Three frozen detectors: original batch-normalized C8741 (512 rows), source-fitted fixed normalization, canonical NLL. No target fitting or selection. Source and target use the same prescribed 9 datasets. Six results are preserved when three datasets are added.

| Model / scope | Original directed / folded | Fixed directed / folded | NLL directed / folded |
| --- | ---: | ---: | ---: |
| source / full | 66.5122 / 68.9655 | 67.7833 / 70.5585 | 65.6439 / 68.7112 |
| source / strict | 66.4723 / 68.9256 | 67.7437 / 70.5189 | 65.6177 / 68.6849 |
| target / full | 62.7792 / 64.9946 | 63.5637 / 66.5841 | 61.4593 / 64.8345 |
| target / strict | 62.7146 / 64.9300 | 63.5060 / 66.5263 | 61.4059 / 64.7811 |

500 deterministic paired claim draws; strict draws use only retained claim groups. Per-dataset folding precedes each fixed-dataset mean. Full paired intervals and matched-transfer differences are in JSON.

Source zero-MAD rules permanently disable f3/f4 on target. This tests a frozen feature/scale combination; medians shift an offset and AUROC does not establish calibration transfer. Exploratory after source benchmark viewing; no original primary is replaced.
