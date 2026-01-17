from __future__ import annotations

import json
from pathlib import Path

import typer
from datasets import load_from_disk

from llm_customization_ops.config import settings
from llm_customization_ops.data.make_pref_dataset import build_pref_dataset
from llm_customization_ops.data.make_sft_dataset import build_sft_dataset
from llm_customization_ops.eval.gates import gate_report
from llm_customization_ops.eval.harness import run_eval
from llm_customization_ops.ops.logging import configure_logging
from llm_customization_ops.training.distill import run_distillation
from llm_customization_ops.training.dpo import run_dpo
from llm_customization_ops.training.registry import (
    get_template,
    list_templates,
    validate_registry,
)
from llm_customization_ops.training.sft_lora import run_sft_lora
from llm_customization_ops.training.sft_qlora import run_sft_qlora

app = typer.Typer(add_completion=False)


@app.callback()
def setup() -> None:
    configure_logging(settings.log_level)


registry_app = typer.Typer()


@registry_app.command("list")
def registry_list(
    path: Path = settings.paths.repo_root / "config" / "prompt_templates.json",
) -> None:
    for template_id in list_templates(path):
        typer.echo(template_id)


@registry_app.command("show")
def registry_show(
    template_id: str,
    path: Path = settings.paths.repo_root / "config" / "prompt_templates.json",
) -> None:
    template = get_template(path, template_id)
    typer.echo(json.dumps(template.model_dump(), indent=2))


@registry_app.command("validate")
def registry_validate(
    path: Path = settings.paths.repo_root / "config" / "prompt_templates.json",
) -> None:
    issues = validate_registry(path)
    if issues:
        typer.echo("\n".join(issues))
        raise typer.Exit(code=1)
    typer.echo("ok")


app.add_typer(registry_app, name="registry")

data_app = typer.Typer()


@data_app.command("make-sft")
def make_sft(input_path: Path, output_dir: Path) -> None:
    build_sft_dataset(input_path, output_dir)
    typer.echo(f"saved {output_dir}")


@data_app.command("make-pref")
def make_pref(input_path: Path, output_dir: Path) -> None:
    build_pref_dataset(input_path, output_dir)
    typer.echo(f"saved {output_dir}")


app.add_typer(data_app, name="data")


train_app = typer.Typer()


@train_app.command("sft")
def train_sft(
    dataset_path: Path = settings.paths.repo_root / "artifacts" / "sft",
    base_model: str = settings.training.base_model,
    output_dir: Path = settings.paths.repo_root / "artifacts" / "runs" / "sft",
    batch_size: int = settings.training.batch_size,
    learning_rate: float = settings.training.learning_rate,
    epochs: int = settings.training.epochs,
    max_steps: int = settings.training.max_steps,
    seed: int = settings.training.seed,
) -> None:
    dataset = load_from_disk(str(dataset_path))
    run_sft_lora(
        dataset,
        base_model,
        output_dir,
        batch_size,
        learning_rate,
        epochs,
        max_steps,
        seed,
    )
    typer.echo("sft complete")


@train_app.command("qlora")
def train_qlora(
    dataset_path: Path = settings.paths.repo_root / "artifacts" / "sft",
    base_model: str = settings.training.base_model,
    output_dir: Path = settings.paths.repo_root / "artifacts" / "runs" / "qlora",
    batch_size: int = settings.training.batch_size,
    learning_rate: float = settings.training.learning_rate,
    epochs: int = settings.training.epochs,
    max_steps: int = settings.training.max_steps,
    seed: int = settings.training.seed,
) -> None:
    dataset = load_from_disk(str(dataset_path))
    report = run_sft_qlora(
        dataset,
        base_model,
        output_dir,
        batch_size,
        learning_rate,
        epochs,
        max_steps,
        seed,
    )
    typer.echo(json.dumps(report, indent=2))


@train_app.command("dpo")
def train_dpo(
    dataset_path: Path = settings.paths.repo_root / "artifacts" / "pref",
    base_model: str = settings.training.base_model,
    output_dir: Path = settings.paths.repo_root / "artifacts" / "runs" / "dpo",
    batch_size: int = settings.training.batch_size,
    learning_rate: float = settings.training.learning_rate,
    epochs: int = settings.training.epochs,
    max_steps: int = settings.training.max_steps,
) -> None:
    dataset = load_from_disk(str(dataset_path))
    run_dpo(
        dataset, base_model, output_dir, batch_size, learning_rate, epochs, max_steps
    )
    typer.echo("dpo complete")


@train_app.command("distill")
def train_distill(
    dataset_path: Path = settings.paths.repo_root / "artifacts" / "distill",
    teacher_model: str = settings.training.base_model,
    student_model: str = "sshleifer/tiny-gpt2",
    output_dir: Path = settings.paths.repo_root / "artifacts" / "runs" / "distill",
    max_steps: int = settings.training.max_steps,
) -> None:
    dataset = load_from_disk(str(dataset_path))
    run_distillation(dataset, teacher_model, student_model, output_dir, max_steps)
    typer.echo("distill complete")


app.add_typer(train_app, name="train")


eval_app = typer.Typer()


@eval_app.command("run")
def eval_run(
    golden_path: Path = settings.paths.repo_root
    / "src"
    / "llm_customization_ops"
    / "eval"
    / "golden"
    / "golden.jsonl",
    output_dir: Path = settings.paths.repo_root / "artifacts" / "eval",
) -> None:
    report = run_eval(golden_path, output_dir)
    typer.echo(json.dumps(report, indent=2))


@eval_app.command("gate")
def eval_gate(
    report_path: Path = settings.paths.repo_root / "artifacts" / "eval" / "report.json",
    thresholds_path: Path = settings.paths.repo_root / "config" / "eval_gates.json",
) -> None:
    failures = gate_report(report_path, thresholds_path)
    if failures:
        typer.echo("\n".join(failures))
        raise typer.Exit(code=1)
    typer.echo("gates passed")


app.add_typer(eval_app, name="eval")


if __name__ == "__main__":
    app()
