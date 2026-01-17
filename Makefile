PYTHON ?= python

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[dev,train,serve]"

lint:
	ruff check src tests
	ruff format --check src tests

type:
	mypy src

test:
	pytest

cov:
	pytest --cov=llm_customization_ops --cov-report=term-missing

train-sft:
	$(PYTHON) -m llm_customization_ops.cli data make-sft --input-path src/llm_customization_ops/data/fixtures/sft.jsonl --output-dir artifacts/sft
	$(PYTHON) -m llm_customization_ops.cli train sft --dataset-path artifacts/sft

train-qlora:
	$(PYTHON) -m llm_customization_ops.cli data make-sft --input-path src/llm_customization_ops/data/fixtures/sft.jsonl --output-dir artifacts/sft
	$(PYTHON) -m llm_customization_ops.cli train qlora --dataset-path artifacts/sft

train-dpo:
	$(PYTHON) -m llm_customization_ops.cli data make-pref --input-path src/llm_customization_ops/data/fixtures/pref.jsonl --output-dir artifacts/pref
	$(PYTHON) -m llm_customization_ops.cli train dpo --dataset-path artifacts/pref

eval:
	$(PYTHON) -m llm_customization_ops.cli eval run
	$(PYTHON) -m llm_customization_ops.cli eval gate

serve:
	LLM_OPS_FAKE_MODEL=1 uvicorn llm_customization_ops.serving.app:app --host 0.0.0.0 --port 8000

docker-serve:
	docker compose -f docker/compose.yml up --build
