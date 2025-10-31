#!/usr/bin/env bash
set -euo pipefail
python -c "from src.utils.plotting import plot_action_distribution; plot_action_distribution('$1')"
python -c "from src.utils.plotting import plot_value_curves; plot_value_curves('$1')"
