from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Paths(BaseModel):
    repo_root: Path = Field(default_factory=lambda: Path.cwd())
    artifacts_dir: Path = Field(default_factory=lambda: Path.cwd() / "artifacts")


class TrainingConfig(BaseModel):
    base_model: str = "sshleifer/tiny-gpt2"
    output_dir: Path = Field(default_factory=lambda: Path("artifacts") / "runs")
    batch_size: int = 1
    learning_rate: float = 2e-4
    epochs: int = 1
    max_steps: int = 10
    seed: int = 42
    smoke: bool = True


class EvalConfig(BaseModel):
    gates_path: Path = Field(default_factory=lambda: Path("config") / "eval_gates.json")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_OPS_", env_file=".env")
    paths: Paths = Paths()
    training: TrainingConfig = TrainingConfig()
    eval: EvalConfig = EvalConfig()
    log_level: str = "INFO"
    otel_endpoint: str | None = None
    fake_model: bool = False


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
