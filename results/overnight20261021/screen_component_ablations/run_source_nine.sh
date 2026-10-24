#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. CPU only; explicit --mode evaluate in child evaluator.
env PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/prepare_screen_component_ablations.py --output results/overnight20261021/screen_component_ablations --source-cache data/restored/mod1-eval-cache --target-cache data/restored/mod1-transfer-cache --check-caches source-nine
env PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/evaluate_parallel.py --workers 2 --cache-dir data/restored/mod1-eval-cache --output results/overnight20261021/screen_component_ablations/evaluations/source_nine --model-id /workspace/mod1-models/llama31 --data-root data/claims --frozen-glob 'results/overnight20261021/screen_component_ablations/frozen/*/frozen_estimator.py' --method-prefix new/screen_ --bootstrap 500 --bootstrap-unit claim
