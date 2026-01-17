from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm_customization_ops.ops.utils import ensure_dir, write_json
from llm_customization_ops.training.callbacks import RunSummary, SummaryCallback


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tokenize(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _format(batch: dict[str, Any]) -> dict[str, Any]:
        text = (
            f"Instruction: {batch['instruction']}\n"
            f"Input: {batch['input']}\n"
            f"Output: {batch['output']}"
        )
        tokens = tokenizer(text, truncation=True, max_length=256)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(_format)


def run_sft_lora(
    dataset: Dataset,
    base_model: str,
    output_dir: Path,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    ensure_dir(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)

    tokenized = _tokenize(dataset, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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

    summary = RunSummary()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[SummaryCallback(summary)],
    )
    trainer.train()

    model.save_pretrained(str(output_dir / "adapter"))
    report = {
        "base_model": base_model,
        "output_dir": str(output_dir),
        "metrics": summary.metrics,
    }
    write_json(output_dir / "summary.json", report)
    return report
