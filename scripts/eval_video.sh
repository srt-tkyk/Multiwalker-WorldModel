#!/usr/bin/env bash
set -euo pipefail
python -m src.evaluate --config configs/default.yaml --ckpt "$1" --save_video --out "$2"
