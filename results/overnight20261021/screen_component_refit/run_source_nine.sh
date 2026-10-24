#!/usr/bin/env bash
set -euo pipefail
env PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/evaluate_parallel.py --workers 2 --cache-dir data/restored/mod1-eval-cache --output results/overnight20261021/screen_component_refit/evaluations/source_nine --model-id /workspace/mod1-models/llama31 --data-root data/claims --frozen-glob 'results/overnight20261021/screen_component_refit/frozen/*/frozen_estimator.py' --method-prefix new/screen_component_logistic --bootstrap 500 --bootstrap-unit claim
