# Architecture

This repository is organized around clear domains: data, training, evaluation, and serving. Each domain has its own module with explicit inputs/outputs and shared utility helpers.

## Key flows
1. **Data pipelines** validate schema and write HF datasets to `artifacts/`.
2. **Training** loads datasets and fine-tunes adapters using LoRA or QLoRA.
3. **DPO** performs preference tuning (RLHF-lite).
4. **Evaluation** runs on a golden set and enforces gates.
5. **Serving** exposes a FastAPI API with metrics and traces.
