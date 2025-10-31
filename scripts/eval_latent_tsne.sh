#!/usr/bin/env bash
set -euo pipefail
python -m src.evaluate --config configs/default.yaml --ckpt "$1" --dump_latents --latents_out "$2"
python -c "from src.utils.tsne import run_tsne; run_tsne('$2')"
