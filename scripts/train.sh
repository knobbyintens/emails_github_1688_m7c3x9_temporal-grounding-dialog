#!/usr/bin/env bash
set -euo pipefail
python -m src.train --config configs/default.yaml --output-dir outputs/temporal-grounding
