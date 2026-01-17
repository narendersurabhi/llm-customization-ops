from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class PromptTemplate(BaseModel):
    version: str
    template: str


def load_registry(path: Path) -> dict[str, PromptTemplate]:
    payload = json.loads(path.read_text())
    return {key: PromptTemplate.model_validate(value) for key, value in payload.items()}


def list_templates(path: Path) -> list[str]:
    return sorted(load_registry(path).keys())


def get_template(path: Path, template_id: str) -> PromptTemplate:
    registry = load_registry(path)
    if template_id not in registry:
        raise KeyError(f"Unknown template_id: {template_id}")
    return registry[template_id]


def validate_registry(path: Path) -> list[str]:
    registry = load_registry(path)
    issues: list[str] = []
    for key, template in registry.items():
        if "{text}" not in template.template:
            issues.append(f"Template {key} missing '{{text}}' placeholder")
    return issues
