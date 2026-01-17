from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


class SFTRecord(BaseModel):
    instruction: str
    input: str
    output: str

    @field_validator("instruction", "output")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value


class PreferenceRecord(BaseModel):
    prompt: str
    chosen: str
    rejected: str

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value


def validate_jsonl(
    payload: list[dict[str, Any]], schema: type[BaseModel]
) -> list[BaseModel]:
    return [schema.model_validate(item) for item in payload]
