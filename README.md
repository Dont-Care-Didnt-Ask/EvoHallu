# EvoHallu

AlphaEvolve-style evolution of **unsupervised uncertainty estimators** for hallucination detection. An LLM writes Python estimators, a labelled training set scores them, and a program database supplies successful examples for subsequent rounds. “Unsupervised” means labels are **not inputs to the estimator**; evolution still uses labels to select programs. No LLM finetuning is performed.

This guide follows the **single-objective atomic-claim task**, with **GPT-OSS-120B writing estimator code** and **Llama-3.1-8B producing the uncertainty inputs**. These are different models: do not replace the preprocessing model with GPT-OSS unless you deliberately want a different experiment.

## Code map

All commands below run from the repository root.


| Step | What to run / edit |
|---|---|
| Precomputation | `scripts/precompute_uq_features_separated.sh` |
| Evolution | `scripts/run_uq_docker.sh`, `configs/config_uq.yaml` |
| Evaluation | Add selected candidates to `found_formulas.py`, import them in `end2end_eval_original.py`, run `scripts/end2end.sh`. |

Supporting code:

- `src/experiments/main.py`: evolution entry point (normally 500 rounds); Docker runs it through `scripts/uq_wrapper.sh`.
- `configs/task/uncertainty_quantification.yaml`: dataset paths, train/validation split and input regime.
- `src/packing/evaluate/uncertainty_quantification/precompute_raw_stats_separated.py`: model forward passes and saved statistics, called by the preprocessing wrapper.
- In that same UQ directory, `task_uncertainty_quantification.py` defines the initial estimator, scoring and task prompt; `features.py` converts raw tensors to estimator inputs; `found_formulas.py` and `end2end_eval_original.py` implement benchmark methods/evaluation.

The remaining `packing` modules mostly implement prompting, model serving, parallel evaluation and the evolutionary program database.

## 0. Environment and data

```bash
cd /mnt/extremessd10tb/seleznev/EvoHallu
conda env create -f local_poly.yml
conda activate local_poly
python -m pip install -e .
# Imported by the evolution framework, but absent from the environment export:
python -m pip install peft
# Skip cloning if an appropriate local checkout already exists.
git clone https://github.com/IINemo/lm-polygraph.git ../lm-polygraph
python -m pip install -e ../lm-polygraph
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

This host Python environment is for **preprocessing and evaluation**; evolution uses the Docker image's environment. `local_poly.yml` is a historical Linux/CUDA export, not a portable lockfile. Use an existing working project environment if available. Evaluation imports custom LM-Polygraph estimators; a stock checkout may not contain them (see step 3).

Obtain the prepared PopQA train/test CSVs from the maintainer (not included here). Each needs `claim` (text) and `is_hallu` (**1 = hallucination/false, 0 = true**). Keep train/test claims disjoint, with both classes present. The default experiment evolves on **PopQA only**.

```bash
export CLAIM_DATA=/absolute/path/to/prepared/claim/csvs  # contains popqa/
# Default persistent cache shared by host preprocessing and Docker evolution:
export HOST_CACHE="$(dirname "$PWD")/.cache/huggingface"
mkdir -p "$HOST_CACHE"
```

Evolution additionally needs Docker access, NVIDIA Container Toolkit, and a GPT-OSS-capable image. The launcher's historical default is `seleznev/seleznev/packing:amd64-cuda-seleznev-latest`; obtain access to it from the maintainer or set `IMAGE` to a compatible image (vLLM plus the project dependencies). Pull it once with `docker pull "$IMAGE"` if using a custom tag. The project image installs the bind-mounted repository on startup; it does not use the host conda environment. See `installation/docker-amd64-cuda-vllm/` for image templates.

The image's user must be able to write to the repository (`out/`) and persistent cache. If needed, authenticate for gated feature-model downloads on the host before preprocessing. Both feature computation and GPT-OSS evolution require sufficient GPU memory.

## 1. Precompute training/test inputs

With the host Python environment active, run:

```bash
bash scripts/precompute_uq_features_separated.sh
```

The wrapper reads `$CLAIM_DATA/popqa/popqa_{train,test}_filtered_10shot.csv` (default data root: `../xx_datasets`) and writes **three** files per CSV under `data/uncertainty_quantification/`: `*_raw_stats_logits.pt`, `*_raw_stats_hidden_states.pt`, and `*_raw_stats_attentions.pt`. Evolution reads only the logits files by default; its task config now points to these PopQA outputs. Preprocessing still computes all three modalities.

Defaults are GPU 0, batch size 16, top-k 512 and every hidden-state layer. Override them without editing Python:

```bash
UQ_DEVICES=1 BATCH_SIZE=4 bash scripts/precompute_uq_features_separated.sh
# Smoke test only (first 32 rows per CSV; rerun without this limit for real inputs):
bash scripts/precompute_uq_features_separated.sh --max-samples 32
```

Other wrapper settings: `MODEL_ID`, `TOP_K`, `LAYER_STRIDE`, `OUTPUT_DIR`, `CACHE_DIR`, and `PYTHON`. Change the task paths too if you change `OUTPUT_DIR`. Extra command-line arguments are forwarded to the Python precompute script.

Key options:

- `--model-id`, `--chat-template`: fix the feature model and prompt. `generate_true_statement` wraps each claim as the assistant response to “Generate true statement”. Keep them consistent across splits and evaluation.
- `--top-k 512`: retain the 512 largest logits and their token IDs at each position, plus full-vocabulary-derived sequence/greedy log probabilities and token entropies. This is **not** a saved full-vocabulary logits tensor.
- `--batch-size 16`: lower to 1–4 for long claims or GPU OOM. The script loads model weights in float32 and accumulates outputs in host RAM before saving.
- `--layer-stride 1`: preserve every hidden-state layer; keep 1 for full-input experiments.
- `--max-samples N`: first N rows only, useful for a smoke test; omit for the actual experiment.



## 2. Evolve with GPT-OSS-120B

Use `configs/config_uq.yaml` (not the generic `config.yaml`). It selects the single-objective UQ task, 500 rounds, two islands and two outputs per prompt.

After step 0's Docker/cache setup and step 1's preprocessing, simply run:

```bash
bash scripts/run_uq_docker.sh
```

The launcher mounts the repository, its parent data directory and persistent HF/vLLM caches at the same paths inside the container. It calls `scripts/uq_wrapper.sh`, which runs `src/experiments/main.py` with `config_uq` and local vLLM enabled. The runner starts the GPT-OSS server itself; **do not start a separate server**. `get_full_model_name()` now uses `openai/gpt-oss-120b`, so it downloads through the shared HF cache rather than another user's hard-coded model path. Outputs remain on the host under `out/`.

Defaults: repository inferred from the script location, `HOST_DATA` = repository parent, `HOST_CACHE` = `$HOST_DATA/.cache/huggingface`, `UQ_DEVICES=0`, `CLUSTER=example`. Set `IMAGE`, `HOST_CACHE`, or `HOST_DATA` if your installation differs; `HOST_CACHE` must already exist. `HOST_VLLM_CACHE` defaults to `$HOST_CACHE/vllm_cache` and is created automatically.

```bash
# Choose a physical GPU; extra arguments override Hydra configuration:
UQ_DEVICES=1 bash scripts/run_uq_docker.sh
# Short smoke test, in a separate run directory:
bash scripts/run_uq_docker.sh num_rounds=2 run_identifier_name=gptoss120b_smoke
```

The default launcher supports **one GPU**. Ensure GPT-OSS-120B fits it. For multi-GPU serving, adapt the wrapper's single-GPU check and `start_vllm_server()` in `src/packing/model/model.py` (device visibility and tensor-parallel/memory options); passing several IDs alone is not sufficient. That server function currently sets `--max-model-len 16384`.

vLLM sampling settings are in `configs/model/gpt-oss-120b.yaml`: `model.temperature=1.0`, `model.max_tokens=4096`, and `model.topp=0.95`. `model.topk` is not sent by the current request function.

Main hyperparameters (`configs/task/uncertainty_quantification.yaml` unless noted):


| Setting                                                         | Meaning / default                                                                                                      |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `task.sample_size=0`                                            | Use all rows before splitting; positive values select a stratified subset.                                             |
| `task.val_size=0.2`, `task.split_seed=441`                      | Deterministic stratified 80/20 train/validation split of the training CSV.                                             |
| `task.input_regime=logits`, `task.supervised=false`             | Expose logit-derived inputs but no labelled reference statistics.                                                      |
| `task.chunk_size=512`                                           | Number of samples passed to an estimator per evaluation chunk. Keep fixed: batch-relative estimators can depend on it. |
| `task.timeout_period=240`, `task.mem_limit_gb=64`               | Candidate evaluation time/memory limits.                                                                               |
| `num_rounds=500`, `num_outputs_per_prompt=2`                    | Evolution length and candidates sampled per prompt; rounds are not individual candidate counts.                        |
| `programdatabaseConfig.num_islands=2`, `functions_per_prompt=2` | Independent program pools and number of examples supplied in a prompt.                                                 |
| `seed=1046`                                                     | Evolution RNG seed; LLM generation may still be nondeterministic.                                                      |


Start with `num_rounds=2` under a **different run name** to check the pipeline.

Outputs go to `out/logs/popqa_logits/uncertainty_quantification_gptoss120b_1046/`:

- `config.yaml`, `stdout.log`, `programbank/`: configuration, log and program-pool snapshots.
- `all_metrics_train.csv`: candidate training metrics; `uq_split_metrics.csv`: train/validation/test metrics for each round's training winner.
- `candidates/candidate_round_*_function_*.txt`: each round's winning estimator source, including imports and metric comments.

Scores are **ROC-AUC × 100**; larger uncertainty must mean more likely hallucination. Although validation/test are evaluated each round, registration in the evolutionary pool uses the training score. Select your final estimator on **validation**, freeze it, and only then report held-out results; do not choose by the test-score comments. Reusing a run directory with `flag_resume.txt` resumes it.

## 3. Evaluate selected estimators on nine claim datasets

The historical list in `scripts/end2end.sh` is:

1. PopQA (`popqa/popqa_test_filtered_10shot.csv`)
2. AVeriTeC (`AVeriTeC/dev_filtered_10shot.csv`)
3. UHEAD (`UHEAD_bio_gemma_annotated_by_mistral/test_filtered_10shot.csv`)
4. Cities (`geometry/cities.csv`)
5. Companies (`geometry/companies_true_false.csv`)
6. Common claims (`geometry/common_claim_true_false.csv`)
7. CounterFact (`geometry/counterfact_true_false.csv`)
8. X-Fact (`x-fact/test_filtered_10shot.csv`)
9. WildHallucinations (`WildHallucinations/pred_PerplexityAI_retrieval+llama_eval_vllm_Llama3_inst_perplexity_filtered_10shot.csv`)

**Canonical workflow: select → add to** `found_formulas.py` **→ import/instantiate → run** `scripts/end2end.sh`**.** Evaluation recomputes model inputs; no separate preprocessing of the nine evaluation datasets is required.

1. **Select the top five distinct candidates by validation AUC across the evolution run**, not by test AUC or just the last round. Use `uq_split_metrics.csv` (`split=val`) and the `# Val AUC` comments in `candidates/*.txt` to rank the saved candidates. Important: these files contain each round's **training winner**, not every generated program. If selecting from the entire generated pool, evaluate the other saved programs on the same validation split first; `all_metrics_train.csv` alone does not provide their validation scores. Freeze the selection before benchmarking.
2. **Copy the selected estimator classes and required imports into** `src/packing/evaluate/uncertainty_quantification/found_formulas.py`. Rename each `HallucinationDetector` to a unique descriptive name, e.g. `GPTOSS120B_Round123_Function1`; preserve its scoring logic and dependencies. Add a unique `__str__()` returning that name, following the existing formulas: the evaluator uses string names as result keys. Record the source run/round and validation score in a comment. Do not copy a second `Estimator` implementation or switch its import namespace.
3. **Edit** `end2end_eval_original.py` in the same directory to import these classes from `found_formulas.py`, then add instances to its `estimators` list, alongside the desired baselines. For example:
  ```python
   # Add to the existing found_formulas import:
   GPTOSS120B_Round123_Function1,
   # Add to estimators (inside main, where context already exists):
   GPTOSS120B_Round123_Function1(context=context),
  ```
   The current active baselines include `SequenceLogProb`, `MeanTokenEntropy`, `RAUQ`, and `AttentionScore`. Its LM-Polygraph imports require the matching custom checkout; remove unavailable imports/methods if they are not part of your comparison. There is **no** `--candidate` **option**, and the wrapper does not automatically read your evolution pool.
4. **Edit** `scripts/end2end.sh`: set `base_dir` to the parent of the `datasets/` directory, `devices` to the evaluation GPU, and `output_dir` to a fresh run-specific name. Keep `--model-id` (default `VityaVitalich/Llama3.1-8b-instruct`), `--chat-template generate_true_statement`, and `--top-k` (default 512) consistent with preprocessing. Lower its per-dataset batch sizes if needed for GPU memory, then run:
  ```bash
   bash scripts/end2end.sh
  ```

The wrapper runs `end2end_eval_original.py` on all nine CSVs and writes metric JSONs and `*_bootstrap_aucs.json` files under `out/logs/example/EVALS/<output_dir>/` (1,000 aligned bootstrap resamples, seed 31). Use a fresh output directory: the evaluator skips a dataset if its metric output already exists. GPT-OSS/vLLM is not needed for this stage, only the feature model.

**Protocol note:** this evaluator builds full inputs even for selected logit-only estimators and truncates claims to **300 characters**. Training preprocessing does not apply that character limit. Keep this behavior to reproduce the historical benchmark; document any change, since complete-claim results are not interchangeable.

## Additional information: changing the workflow

### Use all available inputs instead of only logits

Changing `task.input_regime=full` alone is **not enough**. The single-task loader reads one `.pt` file and does not merge its siblings.

1. Precompute with step 1, retaining hidden states and attentions; keep `--layer-stride 1`. Attention maps grow quadratically with token length, so lower the batch size and plan for substantial RAM/disk usage.
2. Merge the three matching files **sample by sample**, retaining their common metadata. For example, repeat this for train and test (evaluation through `end2end.sh` does not need caches):
  ```bash
   export RAW_BASE=data/uncertainty_quantification/popqa_train_filtered_10shot_raw_stats
   python - <<'PY'
   import os, torch
   base = os.environ['RAW_BASE']
   load = lambda suffix: torch.load(base + suffix + '.pt', map_location='cpu',
                                   mmap=True, weights_only=False)
   merged, hs, attn = load('_logits'), load('_hidden_states'), load('_attentions')
   assert merged['meta'] == hs['meta'] == attn['meta']
   assert len(merged['samples']) == len(hs['samples']) == len(attn['samples'])
   for dst, h, a in zip(merged['samples'], hs['samples'], attn['samples']):
       for src in (h, a):
           assert dst['claim'] == src['claim'] and dst['label'] == src['label']
           assert torch.equal(dst['input_ids'], src['input_ids'])
       dst['hidden_states'], dst['attentions'] = h['hidden_states'], a['attentions']
   torch.save(merged, base + '_full.pt')
   PY
  ```
3. Repeat evolution with **a new prefix/run name**, `task.input_regime=full`, and `task.train_raw_path` / `task.test_raw_path` pointing to the merged `*_full.pt` files. Keep `task.supervised=false` and `background_raw_path` empty for the unsupervised setting. Reduce `task.chunk_size` and/or increase evaluator limits if needed.
4. Use the same canonical evaluation workflow in step 3: add the selected full-input estimators to `found_formulas.py`, import and instantiate them in `end2end_eval_original.py`, then run `scripts/end2end.sh`. That evaluator already builds full inputs; no evaluation caches or additional regime switch are needed.

Full inputs add hidden-state embeddings and attention-map features to the logit-derived features. `features.py` and the task's `system_prompt_fabric()` document their meanings. The task prompt currently describes all feature families even in logits-only runs, so a generated program may request unavailable features and fail evaluation.

### Precomputation / evaluation script status

Paths in this table are relative to `src/packing/evaluate/uncertainty_quantification/` unless prefixed with `scripts/`. “Legacy” describes the current workflow, not a formal deprecation annotation.


| Script                                                                                            | Use / caveats                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `precompute_raw_stats_separated.py`                                                               | Main atomic-claim precompute; three separate monolithic `.pt` files. Usable with suitable model/data/resources; no logits-only switch or checkpointed chunk writing.                                                                                                                                                |
| `scripts/precompute_uq_features_separated.sh`                                                     | PopQA train/test wrapper; configure `CLAIM_DATA`, `UQ_DEVICES` and batch settings through environment variables.                                                                                                                                                                                                                               |
| `precompute_raw_stats_polygraph.py`                                                               | Older LM-Polygraph precompute; prompt + generated-answer inputs, generation-token-only statistics, three separate files. Not a replacement for claim preprocessing.                                                                                                                                                 |
| `precompute_raw_stats_polygraph_chunked.py`                                                       | Prefer for large **LM-Polygraph** inputs: stores all modalities together in `*_raw_stats_chunk_<i>.pt` (`--chunk-size` controls rows/file), or one combined file without chunking. Uses generation quality scores, not claim labels. The single-task `_load_raw()` cannot consume a directory/list of chunks as is. |
| `scripts/precompute_uq_features_polygraph_chunked.sh`, `scripts/parallel_precompute_polygraph.sh` | Machine/path-specific Polygraph launch helpers; inspect/edit before use.                                                                                                                                                                                                                                            |
| `end2end_eval_original.py`                                                                        | Canonical claim evaluator, despite “original” in its name; add selected formulas to its imports/estimator list. Custom LM-Polygraph imports, 300-character truncation.                                                                                                                                              |
| `end2end_eval.py`                                                                                 | Refactored alternative, **not usable as is**: calls undefined `load_model_and_tokenizer`; also has an undefined `function_class` in its bad-shape branch. Do not substitute it for the historical wrapper without fixing it.                                                                                        |
| `src/experiments/eval.py`                                                                         | Generic program-bank evaluation, not the nine-claim benchmark runner; forces W&B and retains generic-framework assumptions. Use `found_formulas.py` + `end2end_eval_original.py` + `scripts/end2end.sh` for selected candidates.                                                                                    |




### The other benchmark family (do not mix protocols)

LM-Polygraph uses `prepare_polygraph_generations.py` → `compute_polygraph_metrics.py` → `precompute_raw_stats_polygraph_chunked.py` for generation/quality/statistics, and `eval_prr_polygraph.py` / `scripts/end2end_prr.sh` for evaluation. Inputs include `input`, `model_generation`, reference `output`, and task-specific quality columns (e.g. F1, ROUGE-L, BLEU, exact match). The objective is **prediction-rejection ratio (PRR)** based on continuous answer quality, not binary claim ROC-AUC. The current PRR wrapper lists **12 datasets**, not the nine claim datasets.

Multi-dataset optimization lives in `task_uncertainty_quantification_multi.py` and `configs/config_uq_multi_train_mix_generations.yaml`; its current objective aggregates mean/minimum PRR. It is outside this reproduction guide. Before using it, read `configs/TRAIN_TEST_REMINDER.md`: its split semantics currently reuse the same underlying data and need repair for a proper train/validation/test experiment.

These instructions are based on source inspection. A full GPU/vLLM reproduction has not been run in this checkout; exact historical tables additionally require the original datasets, candidate selection and compatible dependency/custom LM-Polygraph versions.