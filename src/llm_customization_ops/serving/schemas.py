from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    template_id: str
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    model: str | None = None


class ExtractRequest(BaseModel):
    text: str
    template_id: str = "extraction"
    max_new_tokens: int = 64


class ExtractResponse(BaseModel):
    data: dict[str, Any]
