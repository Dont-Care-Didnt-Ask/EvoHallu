# Source-fixed C8741 normalization transfer

Status: **complete_exploratory_six**.

Three frozen detectors: original batch-normalized C8741 (512 rows), source-fitted fixed normalization, canonical NLL. No target fitting or selection. Source and target use the same prescribed 6 datasets. Six results are preserved when three datasets are added.

| Model / scope | Original directed / folded | Fixed directed / folded | NLL directed / folded |
| --- | ---: | ---: | ---: |
| source / full | 65.9793 / 69.6593 | 68.3712 / 72.5340 | 66.4049 / 71.0058 |
| source / strict | 65.9765 / 69.6565 | 68.3684 / 72.5312 | 66.4024 / 71.0033 |
| target / full | 64.1097 / 67.1301 | 65.8436 / 69.4909 | 63.7298 / 67.7914 |
| target / strict | 64.1067 / 67.1272 | 65.8407 / 69.4880 | 63.7264 / 67.7880 |

500 deterministic paired claim draws; strict draws use only retained claim groups. Per-dataset folding precedes each fixed-dataset mean. Full paired intervals and matched-transfer differences are in JSON.

Source zero-MAD rules permanently disable f3/f4 on target. This tests a frozen feature/scale combination; medians shift an offset and AUROC does not establish calibration transfer. Exploratory after source benchmark viewing; no original primary is replaced.
