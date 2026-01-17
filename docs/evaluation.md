# Evaluation

The evaluation harness reads the golden set at `src/llm_customization_ops/eval/golden/golden.jsonl`, computes simple metrics, and writes a report to `artifacts/eval/report.json`.

```bash
python -m llm_customization_ops.cli eval run
python -m llm_customization_ops.cli eval gate
```

Thresholds are defined in `config/eval_gates.json`.
