#!/usr/bin/env bash
set -euo pipefail
python -m llm_customization_ops.cli data make-sft --input-path src/llm_customization_ops/data/fixtures/sft.jsonl --output-dir artifacts/sft
python -m llm_customization_ops.cli train sft --dataset-path artifacts/sft --max-steps 1
python -m llm_customization_ops.cli eval run
python -m llm_customization_ops.cli eval gate
