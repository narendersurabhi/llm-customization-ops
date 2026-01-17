import os
from pathlib import Path

import pytest
from datasets import Dataset

from llm_customization_ops.training.sft_lora import run_sft_lora


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="slow test")
def test_training_smoke(tmp_path: Path) -> None:
    dataset = Dataset.from_list(
        [
            {"instruction": "Echo", "input": "hello", "output": "hello"},
        ]
    )
    output_dir = tmp_path / "run"
    report = run_sft_lora(
        dataset=dataset,
        base_model="sshleifer/tiny-gpt2",
        output_dir=output_dir,
        batch_size=1,
        learning_rate=2e-4,
        epochs=1,
        max_steps=1,
        seed=42,
    )
    assert "base_model" in report
