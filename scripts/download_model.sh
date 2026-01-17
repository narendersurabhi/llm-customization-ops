#!/usr/bin/env bash
set -euo pipefail
MODEL_NAME=${1:-sshleifer/tiny-gpt2}
python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("$MODEL_NAME")
AutoTokenizer.from_pretrained("$MODEL_NAME")
print("Downloaded", "$MODEL_NAME")
PY
