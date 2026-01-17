from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm_customization_ops.ops.utils import ensure_dir, write_json
from llm_customization_ops.training.callbacks import RunSummary, SummaryCallback
from llm_customization_ops.training.sft_lora import _seed_everything, _tokenize


def _bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def run_sft_qlora(
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
    if not _bitsandbytes_available():
        skip_report: dict[str, Any] = {
            "status": "skipped",
            "reason": "bitsandbytes not available. Install bitsandbytes and ensure GPU support.",
        }
        write_json(output_dir / "summary.json", skip_report)
        return skip_report

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quant_config, device_map="auto"
    )
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
    report: dict[str, Any] = {
        "base_model": base_model,
        "output_dir": str(output_dir),
        "metrics": summary.metrics,
    }
    write_json(output_dir / "summary.json", report)
    return report
