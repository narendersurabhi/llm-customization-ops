from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


def load_model(base_model: str, fake: bool = False) -> LoadedModel:
    if fake:
        return LoadedModel(model=None, tokenizer=None)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LoadedModel(model=model, tokenizer=tokenizer)
