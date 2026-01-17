from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

from llm_customization_ops.ops.utils import ensure_dir, write_json


def run_dpo(
    dataset: Dataset,
    base_model: str,
    output_dir: Path,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    max_steps: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    ref_model = AutoModelForCausalLM.from_pretrained(base_model)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=max_steps,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(str(output_dir / "adapter"))
    report = {"base_model": base_model, "output_dir": str(output_dir)}
    write_json(output_dir / "summary.json", report)
    return report
