# EvoHallu overnight environment reproduction

**Final closeout, 22 October 2026:** RunPod confirmed `EXITED` / `runtime: null` at **00:44:41 UTC** after all scoring/backups. Compute estimate is **$23.16 / 6 h 28 min** at $3.58/hour total, excluding storage/network and pre-task waiting. [Stop receipt](pod_stop_confirmation.json), [resources](resource_summary/RESOURCE_USAGE.md), [scientific backup validation](final_local_backup_validation.json). Earlier dated pending statuses below are historical snapshots superseded by this closeout and the target-completion addendum. No second clean rebuild is claimed.

**The working pod environment is verified; a second clean rebuild has not been performed.** The initial **21 October 2026, 21:43 UTC** read-only audit examined package metadata, installation logs, server receipts and dependency consistency. That audit installed nothing and launched no GPU work; the dated addenda below separately record actual later inference/serving evidence. Versions and original evidence hashes are in [the audit receipt](environment_reproduction_audit.json).

## Actual tested environment

| Item | Observed value |
| --- | --- |
| Host/container userspace | Ubuntu 24.04.5 LTS, x86_64, glibc 2.39; Linux 6.8.0-147-generic |
| Python / environment tool | **CPython 3.12.3**, base `/usr/bin/python3.12`; **uv 0.13.1** |
| GPUs / driver | **2 × A100-SXM4 80 GB**, 81,920 MiB each; **595.91.07** |
| CUDA | Driver advertises **13.2**; tested Torch build uses **13.0** |
| Torch | Module **2.13.0+cu130**; distribution metadata **2.13.0** |
| Serving / model libraries | **vLLM 0.31.0**, **Transformers 5.17.0**, Accelerate 1.15.0, HF Hub 1.33.0 |
| Numeric / fitted-model libraries | NumPy **2.1.2**, SciPy **1.18.1**, scikit-learn **1.9.1**, pandas **3.0.6**, joblib **1.6.0** |
| Trial configuration/client | Hydra **1.3.7**, OmegaConf **2.3.1**, OpenAI SDK **3.31.0** used against local model endpoints |

Read-only `uv pip check --python .venv/bin/python` passes: **all 227 installed distributions are compatible**. Current names/versions match all **227 entries** in [the original freeze](requirements_gpu_frozen.txt), with no additions or version differences.

Runtime proof at the **initial 21:43 UTC snapshot** is substantive: two Qwen3.5-9B replicas have served the recorded generation jobs; cached source-nine and historical target-six evaluation, fitted controls and component ablations completed in this environment. This did not certify every optional model. **Qwen27B and GLM serving, new 1B feature extraction, and its 427-row current-Torch numerical preflight were pending at that initial snapshot; later status is below.** Six historical 1B caches were scored on CPU; that is separate from generating the three missing caches under the current runtime.

**Later runtime addendum, 21:49 UTC:** the separate [427-row current-runtime 1B pilot](early_numeric_reference/RESULT.md) passed: maximum token-logprob difference **0.00390625**, canonical-NLL Spearman **0.9999998844**, all top-512 logits/IDs exact, and all six historical cache hashes unchanged. This adds direct 1B inference evidence; full missing-three extraction and optional-writer checks remain separate, and the formal preparer still reruns its own numerical guard.

**Later runtime addendum, approximately 23:50 UTC:** both **Qwen3.8-27B original/diverse seed-2107 jobs executed all 60 requests** and have complete source-nine evaluations, **71.155 / 72.306 folded** respectively. Actual [27B server metadata](../../out/overnight20261021/qwen27b_servers.json) records two BF16 replicas with compilation/graphs enabled; this is a successful writer comparison in the same working environment, with one matched seed. [Program/selection audit](WRITER27B_CANDIDATE_INSPECTION.md).

**At the historical 23:50 UTC snapshot, GLM-4.7-Flash had successfully started and was serving actual pilot requests**, using two **BF16**, **eager** replicas with **`TRITON_MLA`** attention on the A100s. Both actual server commands and the ready status are saved in [GLM server metadata](../../out/overnight20261021/glm47_servers.json); [late-controller state](late_extensions_state.json) records the stage events. Completed pilot quality, Devstral execution and full target-nine were **pending then**, superseded by the terminal-writer addendum below. The separate source-fixed normalization **target-six CPU** follow-up was complete with unchanged fitted state; it adds no model-inference or clean-rebuild claim. [Transfer audits](normalization_transfer/INTERPRETATION.md).

**Terminal writer addendum — 22 October 2026, approximately 00:06 UTC (supersedes the 23:50 pending pilot status):** both GLM jobs completed all **60 requests**, original **28 valid** and diverse **36 valid**, with source-nine directed/folded scores **68.533/70.671** and **65.644/68.711**. The diverse winner is exact canonical NLL. Across all writers, **27 jobs / 1,500 requests / 841 valid candidates** completed their planned budgets. GLM's measured-runtime second-seed gate did not fit, and **Devstral was skipped before any serving attempt**: 65.51 minutes remained against its unchanged 75-minute entry rule. [Terminal writer receipt](late_writer_decision.json). This establishes actual BF16 eager/TRITON_MLA GLM execution, not Devstral compatibility or a second clean rebuild.

Full target-nine extraction/scoring and local cache backup remain **PENDING** at this addendum. The late controller refused its immediate post-GLM-cleanup idle check at 00:04:29. After independently verifying two idle snapshots with zero GPU processes/memory use, the root launched the **unchanged formal preparer** at 00:05:48, keeping the 35-minute cap, original 427-row numerical guard and absolute deadlines. The initial refusal is retained; no scientific code, tolerance or source cache was changed to permit the launch. [Archived one-off orchestration](root_transfer_recovery_launch.py), [launch/telemetry receipt](root_transfer_recovery_launch.json). A launch or earlier standalone pilot does not satisfy the formal completion/numerical-reference receipt.

**Target runtime/completion addendum — 22 October 2026, approximately 00:42 UTC (supersedes the preceding pending transfer status):** the unchanged formal preparer completed CounterFact/PopQA/X-Fact at **00:20:50**, **41,253 rows**, with pinned target/shared-tokenizer assets and all six historical raw-cache hashes unchanged. Its repeated **427-row** current-runtime guard passed: observed-token logprob maximum difference **0.00390625**, canonical-NLL Spearman **0.9999998844**, exact row/token/shape/dtype identity and finite raw values. The three lossless XZ archives were verified remotely and locally by **00:30:19**, **762,346,556 bytes**. [Formal completion](remaining_transfer_completion.json), [formal numerical reference](remaining_transfer_numerical_reference.json), [local backup](cache_backup_receipt.json). This is direct 1B extraction evidence in the recorded working environment, still not a second clean rebuild or bitwise NLL equivalence.

Main nine-dataset CPU scoring finished all 18 dataset/kind tasks, retaining **one estimator-level failure** on target UHEAD: **29/30 methods have full-nine results; one has eight**, so no complete three-seed compositional target result is claimed. Saved warnings and frozen source suggest a float16 entropy `0 × log(0)` issue; it was not repaired or newly reproduced. Cache provenance/raw numerical checks and estimator robustness are separate gates. The separately frozen **component-head nine report** and **source-fixed normalization nine report** both have `complete_exploratory_nine` status and exact reference parity, with the original six reports preserved. [Head-nine report](screen_component_refit/transfer_nine/TRANSFER9_REPORT.md), [normalization-nine interpretation](normalization_transfer/NINE_INTERPRETATION.md). Costs and actual pod-stop confirmation remain pending in this snapshot.

The local analysis checkout is a **different CPU environment**: Python 3.12.3, Torch **2.14.1+cpu**, NumPy **2.5.4**, with the same SciPy/scikit-learn/pandas versions above. Saved-NPZ analysis and focused tests ran there. Do not describe it as the pod environment or assume bitwise raw-feature/rescoring parity across these Torch/NumPy versions.

## Reconstructing the observed GPU package set

[requirements-overnight-gpu.txt](../../requirements-overnight-gpu.txt) copies every observed distribution pin unchanged, with no editable/local paths or URLs. It targets the tested Linux/Python/CUDA platform, not arbitrary operating systems. The original freeze remains untouched.

On a **fresh checkout and empty environment** on a matching GPU host, with uv 0.13.1 and Python 3.12.3 already provisioned:

```bash
uv venv --python /usr/bin/python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-overnight-gpu.txt
uv pip check --python .venv/bin/python
PYTHONPATH=src:. .venv/bin/python -c 'import torch,transformers,vllm; assert torch.__version__ == "2.13.0+cu130"; assert torch.version.cuda == "13.0"; print(torch.__version__,transformers.__version__,vllm.__version__)'
```

These are reconstruction commands using the **exact resolved package set in `install_gpu.log`**, rather than a claim that its original shell invocation was recorded. That log starts with Python 3.12.3 and records installation of 227 packages. `venv_setup.log` instead records an earlier **3.11.17** attempt; active `pyvenv.cfg`, `/usr/bin/python3.12` and running imports establish the successful **3.12.3** environment. Use the latter.

The snapshot does not pin wheel hashes, package-index configuration or a container image digest. Torch's plain distribution pin alone does not encode `+cu130`; the build assertion above checks this material distinction. No Docker image is invented or certified here. A second fresh installation/resolution test is still outstanding. The new driver ran the tested CUDA-13 build directly; the previous campaign's CUDA forward-compatibility workaround is not evidence that this pod needed it.

Project imports used **`PYTHONPATH=src:.`**, not an editable `packing` distribution. The uv environment has no `pip` module; use `uv pip`, as above. These commands do not install LM-Polygraph: the tested bounded trial/cached evaluator imports the repository's estimators directly. The historical LM-Polygraph end-to-end evaluation path is a separate setup.

## Writer identity and actual serving profile

The following table preserves the **initial 21:43 UTC audit status**; its pending 27B/GLM entries are superseded by the dated 23:50 runtime addendum above.

| Checkpoint | Pinned revision | Coverage at initial 21:43 audit |
| --- | --- | --- |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Actual two-replica writer runtime |
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Downloaded/prepared; GPU comparison pending |
| zai-org/GLM-4.7-Flash | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | Optional prepared writer; serving/quality pending |
| Llama3.1-8b-instruct source feature checkpoint/tokenizer | `247cb7c3baebabc46cff1fb97a3a5913e4c4511d` | Historical source feature caches; shared tokenizer retained |
| unsloth/Llama-3.2-1B-Instruct target | `5a8abab4a5d6f164389b1079fb721cfab8d7126c` | Six historical caches; current-runtime missing-three extraction pending |

Download manifests retain individual file SHA256s. **Qwen3.5-9B's** tested servers use **BF16, 16,384 context, two concurrent sequences per replica, GPU-memory fraction 0.70**, ports **18000/18001**, with text-only loading and the `qwen3` reasoning parser. Requests set `enable_thinking=false`. **Compilation/graphs are enabled**: the inherited `trial.serving_profile: vllm_eager` label is stale. [Actual server commands](serving_profile_clarification.json) are authoritative; later 27B/GLM serving profiles are recorded separately above.

Fresh-pod reconstruction of that serving setup, after dependencies and training data/caches are available:

```bash
.venv/bin/python scripts/download_checkpoint.py Qwen/Qwen3.5-9B /workspace/models/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
PYTHONPATH=src:. .venv/bin/python scripts/overnight_start_servers.py --model-path /workspace/models/Qwen3.5-9B --ports 18000 18001 --tag qwen9b_ready --gpu-memory-utilization 0.70
PYTHONPATH=src:. .venv/bin/python src/experiments/main.py --config-name next_experiments/next_prompt_original_2107 cluster.api_base_url=http://127.0.0.1:18000/v1 trial.max_seconds=5400
```

The final line is one recorded trial configuration, not the complete campaign scheduler. Qwen writes detector code; it does not replace the Llama source features or generate a new benchmark dataset.

## Historical setup differences that matter

- **`local_poly.yml` is not this environment:** Python 3.11.14, Torch 2.9.1, Transformers 4.50.0, NumPy 1.26.0 and HF Hub 0.36.0 contradict the tested versions. Applying it would downgrade/change the runtime; it is not evidence that the new writers work there.
- **`requirements-trial.txt` is unpinned and omits vLLM.** It is a convenience dependency list, not a serving lock or reproduction of tonight's resolved set.
- **`pyproject.toml` declares no runtime dependencies.** `pip install -e .` alone does not assemble this serving/scoring environment; its Python `>=3.10` declaration is broader than the interpreter actually tested.
- **The README's Docker/GPT-OSS launcher is a different execution route.** Its historical image/default single-GPU wrapper and external LM-Polygraph setup do not reproduce the two direct vLLM replicas above. No second container build was tested here.
- **Cached and regenerated features are distinct evidence.** Historical source/target caches retain their original bytes and float16 storage. Current-runtime 1B additions require pinned assets and the predeclared 427-row numerical guard; package compatibility alone does not establish feature equivalence.

## Offline CPU reuse

Saved JSON and score/bootstrap NPZs can be inspected offline without model weights, an endpoint, GPU or dataset CSVs. **The full report-replay helpers below still require the same ten supplied CSVs under `data/claims/`, in addition to saved artifacts.** In particular, `overnight_report.py` reads them for `baseline_exact_check`; `overnight_component_refit_report.py` also validates CSV hashes, row identities and training-claim overlap. Those CSVs are external to the code/evidence ZIP. On an offline copy with the required artifacts **and CSVs**, for example:

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/overnight_report.py --manifest results/overnight20261021/analysis_manifest.json
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/overnight_component_ablation_report.py --plot
```

These recompute reports from saved predictions and their required CSV identity references, not model features. Do not rerun the sealed component `REFIT_REPORT.*` merely to inspect it; its later target-nine report is separate. Raw-cache rescoring is a separate CPU task and should use the recorded numeric versions or verify exact NLL/row/bootstrap reference parity. It must use **`evaluate_cached_claims.py --mode evaluate`**, the explicit frozen source, unchanged **512-row** scoring chunks and **500 claim** bootstraps; `--mode both` would also extract features. Preserve `is_hallu` and the original float16 NLL accumulation.

Cache/artifact layout:

```text
data/claims/                                      # nine held-out CSVs + PopQA train
data/uncertainty_quantification/                   # pinned PopQA training raw-stat .pt
data/feature_cache_archive/mod1-eval-cache/         # nine .pt.gz archives
data/feature_cache_archive/mod1-transfer-cache/     # six historical .pt.gz archives
data/feature_cache_archive/overnight20261021/       # three new .pt.xz, only if completed/backed up
data/restored/mod1-eval-cache/                      # nine restored raw .pt caches
data/restored/mod1-transfer-cache/                  # historical six + conditional new three
results/overnight20261021/evaluations/              # per-dataset scores/bootstrap NPZs
results/overnight20261021/screen_component_refit/   # frozen weights/pool/helper references
```

`overnight_restore_caches.py --groups mod1-eval-cache --workers 2` restores the nine audited gzip archives on CPU when all archives are present. They expand from **5.95 GiB to 14.76 GiB**; historical target-six expands from **3.41 GiB to 8.48 GiB**, before training caches and working storage. The largest source file is approximately 7 GiB. The current small local filesystem cannot restore the full set; saved-NPZ reporting is the practical local path. New XZ archives have separate lossless/hash receipts and are not automatically covered by the old gzip-only restoration script.

Evidence: [install log](install_gpu.log), [superseded initial venv log](venv_setup.log), [GPU/driver record](nvidia_setup.txt), [freeze](requirements_gpu_frozen.txt), [serving correction](serving_profile_clarification.json), checkpoint/download manifests, exact baseline-reproduction receipts, and the current read-only dependency/metadata audit. No active environment, historical requirement file, frozen helper or scientific manifest was changed by this audit.
