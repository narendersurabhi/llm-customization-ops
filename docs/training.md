# Training

## LoRA
```bash
python -m llm_customization_ops.cli data make-sft --input-path src/llm_customization_ops/data/fixtures/sft.jsonl --output-dir artifacts/sft
python -m llm_customization_ops.cli train sft --dataset-path artifacts/sft
```

## QLoRA
```bash
python -m llm_customization_ops.cli train qlora --dataset-path artifacts/sft
```
If `bitsandbytes` is unavailable, the run will exit with a summary indicating how to enable it.

## DPO
```bash
python -m llm_customization_ops.cli data make-pref --input-path src/llm_customization_ops/data/fixtures/pref.jsonl --output-dir artifacts/pref
python -m llm_customization_ops.cli train dpo --dataset-path artifacts/pref
```

## Distillation
```bash
python -m llm_customization_ops.cli train distill --dataset-path artifacts/distill
```
