from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from llm_customization_ops.ops.utils import ensure_dir, write_json


def run_distillation(
    dataset: Dataset,
    teacher_model: str,
    student_model: str,
    output_dir: Path,
    max_steps: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(teacher_model)
    student = AutoModelForCausalLM.from_pretrained(student_model)

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        tokens = tokenizer(batch["text"], truncation=True, max_length=256)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(_tokenize)

    def _loss_fn(model, inputs, return_outputs=False):
        with torch.no_grad():
            teacher_outputs = teacher(**inputs)
        student_outputs = model(**inputs)
        loss = torch.nn.functional.mse_loss(
            student_outputs.logits, teacher_outputs.logits.detach()
        )
        return (loss, student_outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        max_steps=max_steps,
        logging_steps=1,
        report_to=[],
    )

    trainer = Trainer(
        model=student, args=args, train_dataset=tokenized, compute_loss=_loss_fn
    )
    trainer.train()
    student.save_pretrained(str(output_dir / "student"))
    report = {"teacher": teacher_model, "student": student_model}
    write_json(output_dir / "summary.json", report)
    return report
