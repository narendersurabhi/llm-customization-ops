# Agent Change Log

## 2024-01-01
- Bootstrapped repository structure with src/, tests/, docs/, docker/, scripts/, and config/ directories.
- Added core Python modules for configuration, data pipelines, training (LoRA/QLoRA/DPO/distill), evaluation, serving, and ops utilities.
- Added fixtures, prompt template registry, eval gates, and golden datasets for offline workflows.
- Implemented FastAPI service with metrics, telemetry hooks, and structured logging.
- Added CI workflows, Docker assets, Makefile, pre-commit config, and documentation.

## 2024-01-02
- Fixed lint and type-check issues, updated formatting, and aligned Ruff configuration.
- Added Black to dev dependencies and cleaned import ordering/line lengths.
- Addressed mypy findings in training callbacks and QLoRA reporting paths.
