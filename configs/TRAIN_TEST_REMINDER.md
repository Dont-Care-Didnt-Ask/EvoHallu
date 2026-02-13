# TRAIN/TEST split reminder for UQ multi train-mix variation

Current multi-dataset config `task/uncertainty_quantification_multi_train_mix_generations.yaml`
uses only test-like raw files from:

- `data/uncertainty_quantification/train_mix_generations/*_raw_stats_logits.pt`

Important:

- `task_uncertainty_quantification_multi.py` currently ignores split semantics in `generate_input(...)`.
- `evalset` and `testset` therefore point to the same underlying per-dataset raw files.
- Proper train subsets for `coqa`, `samsum`, `triviaqa`, `wmt14_deen` are still needed for
  a true train/validation workflow.

When train subsets are ready:

1. Extend task config schema to include per-dataset train/val/test paths.
2. Implement deterministic per-dataset split logic in `generate_input`.
3. Ensure no overlap between train and test samples.
4. Re-run preflight checks before optimization.
