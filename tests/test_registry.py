from pathlib import Path

from llm_customization_ops.training.registry import (
    get_template,
    list_templates,
    validate_registry,
)


def test_registry_list() -> None:
    path = Path("config/prompt_templates.json")
    templates = list_templates(path)
    assert "summarization" in templates


def test_registry_show() -> None:
    path = Path("config/prompt_templates.json")
    template = get_template(path, "classification")
    assert "{text}" in template.template


def test_registry_validate() -> None:
    path = Path("config/prompt_templates.json")
    issues = validate_registry(path)
    assert issues == []
