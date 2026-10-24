# Technical review artifacts

These files preserve the experiment audit completed on 19 October 2026.
The patches describe the supplied ZIP versus the earlier experiment snapshot;
they intentionally predate later report, README, and packaging updates.

`selection_audit.json` and `.csv` cover all 15 LLM runs and 676 requests.
The population replay matched 335 recorded prompts and island choices.
`top_validation_candidates.csv` lists up to five saved candidates per run.
`frozen_estimators/` contains all 23 frozen programs, available configurations,
run summaries, and the original exclusion markers for two failed attempts.
The final cohort includes 21 of these frozen programs.

`full_code.patch` contains 57 changed source/configuration/test files.
`full_repository.patch` also includes the historical reports and result files.
Both were verified against a separate copy of the supplied archive.
See `diff_manifest.json` for exact snapshot boundaries and checksums.

Current primary target labels are `is_hallu`; the Sequence NLL baseline is
65.64 macro AUROC. Historical alternative-label diagnostics do not supersede
the primary results in `evohallu_preliminary_report.md`.
