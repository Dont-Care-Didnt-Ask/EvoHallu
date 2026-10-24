# Packaging and publishing the preliminary experiments

This guide preserves the preparation checklist and safe private-repository publishing instructions. Final experiment scoring and backups are complete; RunPod confirmed shutdown at 22 October 00:44:41 UTC. The final package manifest records the actual local commit/archive identities. A portable ZIP excludes `.git`; overlay it onto a fresh clone/new branch using the instructions below, with Vladimir as author. No push or remote branch modification was performed.

## Branch and authorship

The inspected local checkout is on `vladimir/evohallu-preliminary`, with Git identity `Vladimir Urvantsev <urvantsevve@gmail.com>`. At inspection time its committed HEAD was `52ff26004fa884c3ff3f80bcdb8d6f8dc41c7264`; new overnight work was still uncommitted. The local `main` reference was `45dff51361a6f252a17ff71f479bbfb9ea4c5431`. Neither the experiment work nor the publishing instructions require updating `main` or `hallu3`.

The configured private remote is `https://github.com/Dont-Care-Didnt-Ask/EvoHallu.git`. A configured URL does not establish access or compatible Git ancestry. This checkout began from the supplied ZIP: commit `49504d5c89b64288db0b6ef2fe6db7ce8b5d527d` imported that archive. Its author is the historical `Codex <codex@openai.com>` identity; subsequent prepared preliminary commit `52ff260` is Vladimir's. Configuring Vladimir for the next commit does not rewrite previous authors. Importing the final tree into a fresh clone, as below, creates a new Vladimir-authored change on the real repository's history without importing the ZIP's synthetic history.

Before a final local commit, confirm:

```bash
git branch --show-current
git config user.name 'Vladimir Urvantsev'
git config user.email 'urvantsevve@gmail.com'
git rev-parse main
git status --short
```

Do not merge the experiment branch into `main`, reset `main`, or force-push to reconcile different histories.

## Keep code and compact evidence together

The ordinary source snapshot contains `src/`, `scripts/`, `configs/`, `tests/`, root documentation and requirement files, the final English report, figures, existing historical result tables, and the new `results/next_experiments/` and `results/overnight20261021/` trees. Keep immutable scientific manifests and frozen programs byte-identical; changing their paths or reformatting JSON can invalidate recorded hashes. Historical commands may contain the original pod paths; map the new checkout/cache locations through launcher arguments rather than editing the evidence.

The new result tree contains the actual frozen sources, grouped-head weights, component pool, per-dataset score/bootstrap NPZs, duplicate masks, row-group IDs, execution states, and report tables. These are compact experiment artifacts, distinct from model checkpoints and raw feature tensors. Preserve both complete and partial run receipts; a prepared manifest does not establish that a run executed.

The following ignored paths are the minimum additional evidence required by the current manifests and audit scripts. A plain `git add .` omits `out/` and the cache manifest.

| Path | Why it belongs in the compact snapshot |
| --- | --- |
| `out/logs/research20261021/` | Every new run's candidate ledger/code, complete request/round counts, original prompts and responses, resolved redacted trial configuration, selected program, top-three exports, train/validation predictions, and sampler/interface provenance. Keep all seeds, controls and failed/partial attempts. |
| `out/research20261021/supervised_grouped/` | The actual documented-14 `models.joblib` and `results.json`; source and transfer evaluators reference their hashes. This is a roughly 7 KB fitted artifact, not LLM weights. |
| `out/overnight20261021/` | Actual writer server commands/revisions/profiles and GPU telemetry. Preserve terminal receipts elsewhere too; old server PIDs are historical evidence. |
| `out/campaign/evaluation_with_4b/*.json`, `*_scores.npz`, `*_bootstrap.npz` | Historical canonical NLL/labels/row-group/mask and bootstrap references used by the exact nine-dataset baseline comparison. |
| `out/campaign/reference_frozen/` | All **16 historical frozen Python programs** referenced by the saved comparison cohort. Preserve the directory at its recorded relative path. |
| `out/logs/mod1/uncertainty_quantification_{qwen35_4b_evo_1048,qwen35_9b_evo_1047,qwen35_9b_independent_1048,qwen35_9b_seeded_1048}/frozen_estimator.py` | Four additional historical frozen sources. Together with the 16 reference programs and the warm-context 9B/1048 winner below, these supply the **21 historical source paths** referenced by the comparison evidence. All four exact files were checked to exist. |
| `out/logs/mod1/uncertainty_quantification_qwen35_9b_evo_1048/` | Provenance of the 15 inherited parent contexts. `frozen_contexts.json` embeds the chats and hashes, but the original uniformly sampled `prompt_*.json` files establish their source. Keeping the whole roughly 2 MB run also preserves its baseline predictions and ledger. |
| `out/campaign/model_provenance/Llama3.2-1B/`, `out/campaign/model_provenance/llama31/` | Historical target/shared-tokenizer download manifests, configs and template evidence referenced by the pinned current-runtime extraction guard. These directories contain metadata, not checkpoint shards. |
| `out/campaign/transfer_llama32_1b/protocol.json`, `manifest.json` | Historical target extraction/tokenizer protocol used to validate the six original caches. |
| `out/campaign/supervised/`, `supervised_documented/`, `supervised_group/` | Small historical fitted artifacts retained for the evaluator's default provenance scan. Keep their metadata with their `.joblib` files. |
| `data/cache_backup_manifest.json` | Original gzip archive identities and checksums required by restore/transfer provenance. Keep this JSON, not the multi-gigabyte archives, in Git. |

Within `results/overnight20261021/`, keep the full compact trees for `evaluations/`, `frozen/`, `transfer_frozen/`, `transfer_llama32_1b/`, `screen_component_ablations/`, `screen_component_refit/` including its separate `transfer_nine/`, `shortlist_secondary/`, `batch_sensitivity/`, `frozen_train_normalization/`, and the convergence/parent-sampling/failure audits. Preserve the original pinned `REFIT_REPORT.*`; the nine-dataset extension has its own report. Keep all design/selection/activation/validation receipts, final shutdown and cache-backup receipts, and model download receipts. The final report and machine tables should point to the completed evidence, not replace it.

After reviewing the final tree, the following are example staging commands on the Vladimir branch. They are instructions for the final operator, not actions performed by this document:

```bash
git add --update
git add -- src scripts configs tests results *.md *.html requirements-overnight-gpu.txt
git add -f -- out/logs/research20261021 out/research20261021/supervised_grouped out/overnight20261021
git add -f -- out/campaign/evaluation_with_4b/*.json out/campaign/evaluation_with_4b/*_scores.npz out/campaign/evaluation_with_4b/*_bootstrap.npz
git add -f -- out/campaign/reference_frozen
git add -f -- out/logs/mod1/uncertainty_quantification_{qwen35_4b_evo_1048,qwen35_9b_evo_1047,qwen35_9b_independent_1048,qwen35_9b_seeded_1048}/frozen_estimator.py
git add -f -- out/logs/mod1/uncertainty_quantification_qwen35_9b_evo_1048
git add -f -- out/campaign/model_provenance/Llama3.2-1B out/campaign/model_provenance/llama31
git add -f -- out/campaign/transfer_llama32_1b/protocol.json out/campaign/transfer_llama32_1b/manifest.json
git add -f -- out/campaign/supervised out/campaign/supervised_documented out/campaign/supervised_group
git add -f -- data/cache_backup_manifest.json
```

`results/**/*.csv` is explicitly allowed by the current `.gitignore`. Dataset CSVs remain excluded. `.log` files are also ignored: the ZIP should retain the compact controller/install/test/evaluator logs, and Git can selectively retain those cited by reports. For example, after confirming the final files exist:

```bash
git add -f -- results/overnight20261021/controller.log results/overnight20261021/followups.log results/overnight20261021/late_extensions.log
git add -f -- results/overnight20261021/install_gpu.log results/overnight20261021/protocol_tests.log results/overnight20261021/combined_new_protocol_tests.log
git add -f -- results/next_experiments/execution_logs
```

Before committing, inspect `git diff --cached --stat`, `git diff --cached --check`, and the staged filename list. Ensure it contains the frozen programs, `.npy`/`.npz` predictions and small fitted weights, and excludes the items below. Do not use a blanket `git add -f out data results`.

## Keep external data and large assets separate

Exclude these from the code/evidence ZIP and Git commit:

- `.venv/`, `.git/` when making a portable tree ZIP, `__pycache__/`, `.pytest_cache/`, `.cache/`, and temporary/lock/partial-download files.
- Credentials: `*.env`, `configs/cluster/api.yaml`, SSH private keys, account tokens and unredacted environment dumps. New `trial_config.json` files deliberately redact the endpoint key.
- User-supplied dataset CSVs under `data/claims/`, the original dataset ZIP, and raw training `.pt` tensors under `data/uncertainty_quantification/`.
- `data/restored/`, all `.pt` feature caches, `data/feature_cache_archive/` gzip/XZ files, downloaded checkpoint folders and `.safetensors` weight shards. Keep their manifests/checksums and restoration instructions.
- **`results/overnight20261021/early_numeric_reference/scratch_anchor/`**: the separate roughly 87 MiB recomputed AVeriTeC raw cache. Keep its compact pilot/numerical-reference receipts, metadata, log and `RESULT.md`; the raw scratch is unnecessary for the main dataset cache backup.
- Hidden rsync partials, including the inspected historical `out/campaign/evaluation_with_4b/.WildHallucinations__pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot_scores.npz.iZbfIO`. Use canonical `*_scores.npz` files instead. Exclusion does not require deleting the source partial.

The code/evidence ZIP contains saved JSON/NPZ results that can be inspected without dataset CSVs, model weights or inference. **The provided report-replay commands still require the same ten supplied CSVs under `data/claims/`**: `overnight_report.py` reads them for `baseline_exact_check`, and `overnight_component_refit_report.py` checks their hashes, row identities and training-claim overlap. The ZIP deliberately excludes those CSVs, so it alone is not sufficient to rerun these full reporting audits. Raw-feature rescoring additionally needs the audited feature caches. Full regeneration additionally needs the pinned checkpoint/tokenizer assets and recorded runtime. Those are separate levels of reproduction, not properties of a small source ZIP.

### Archived orchestration is specific to this allocation

The one-night orchestration scripts intentionally pin the authorized pod, checkout roots and absolute UTC deadlines for **21–22 October 2026**. Their immutable plans also retain historical absolute paths, often under `/workspace/mod1`. A portable code/evidence ZIP does **not** mean that these archived `--execute` commands should be rerun unchanged on a later allocation. Their guards may correctly refuse another host or an expired window.

Saved JSON/NPZ results can be reviewed offline; running the full report/audit helpers also requires their external CSV prerequisites above. Raw-feature replay needs the documented CSV/cache/checkpoint assets and a recorded mapping to the new filesystem; use explicit supported path arguments or recreate the expected layout without rewriting the historical evidence. A new future experiment needs a **new dated plan/configuration, resource authorization and execution receipts**, rather than edits that make the old campaign appear newly executable. The core trial and preparation framework remain reusable. The observed environment was audited and exercised; a second clean rebuild was not performed or certified. See [environment reproduction](results/overnight20261021/ENVIRONMENT_REPRODUCTION.md) for the tested versions and remaining reconstruction limits.

Keep the external cache bundle independently: nine original source gzip archives, six historical target gzip archives, the training gzip, and three new target XZ archives **only if the final completion and local backup receipts pass**. The original archive manifest and the new lossless raw/compressed SHA256 receipts establish byte identity. The local training raw file was already removed after a verified gzip round trip; its retained archive is `data/feature_cache_archive/train/popqa_train_filtered_10shot_raw_stats_logits.pt.gz`. Restore it only when needed, following `results/overnight20261021/local_storage_preservation.json`.

At the 22:48 inspection, new run logs occupied about 57 MiB; `results/overnight20261021/` occupied about 148 MiB including 87 MiB scratch. These are intermediate sizes, not final ZIP sizes. Write a final package inventory with file SHA256s, selected Git branch/commit, archive-import base and exclusions. Verify ZIP integrity and a clean extraction before sharing it. A historical patch under `results/technical_review/` predates this night; label any new full-code diff with its final source boundaries rather than presenting the old patch as current.

## Publish into the actual private repository safely

If the unpacked ZIP's Git ancestry is uncertain, use a fresh authenticated clone of the real `hallu3` branch. Substitute the actual unpacked snapshot path below. These commands create only the named new branch; the final push is for the user who has private-repository access.

```bash
git clone --branch hallu3 --single-branch https://github.com/Dont-Care-Didnt-Ask/EvoHallu.git EvoHallu-vladimir
cd EvoHallu-vladimir
git switch --create vladimir/evohallu-preliminary origin/hallu3
git config user.name 'Vladimir Urvantsev'
git config user.email 'urvantsevve@gmail.com'
```

Copy the prepared **code/evidence snapshot**, preserving the destination `.git` directory. A safe overlay example is:

```bash
rsync -a --exclude='.git/' --exclude='.venv/' --exclude='.cache/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.env' --exclude='configs/cluster/api.yaml' --exclude='data/claims/' --exclude='data/restored/' --exclude='data/uncertainty_quantification/' --exclude='data/feature_cache_archive/' --exclude='results/overnight20261021/early_numeric_reference/scratch_anchor/' /absolute/path/to/unpacked/mod1/ ./
```

This overlay uses no `--delete`, so upstream-only files remain present. Review the supplied final code diff for any intended deletions and inspect the resulting diff before committing. The overlay excludes the ZIP's `.git`, so it retains the real repository ancestry and does not import synthetic commits. If the proposed branch name already exists locally or remotely, choose a new unused name instead of overwriting it.

Stage the reviewed source/evidence paths above, then:

```bash
git diff --cached --stat
git diff --cached --check
git commit -m 'Add EvoHallu preliminary experiment results'
git log -1 --format='%h %an <%ae> %s'
git branch --show-current
git push --set-upstream origin HEAD:refs/heads/vladimir/evohallu-preliminary
```

The explicit destination pushes only the new experiment branch. It neither pushes `main` nor pushes `hallu3`, and uses no force option. Creating or merging a pull request is a separate user action.
