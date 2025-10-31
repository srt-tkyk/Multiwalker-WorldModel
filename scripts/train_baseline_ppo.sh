#!/usr/bin/env bash
set -euo pipefail
python -m src.train_baseline_ppo --config configs/default.yaml "$@"
