from __future__ import annotations

from fastapi import FastAPI

from llm_customization_ops.config import get_settings
from llm_customization_ops.ops.logging import configure_logging
from llm_customization_ops.serving.model_loader import load_model
from llm_customization_ops.serving.routes import router
from llm_customization_ops.serving.telemetry import init_telemetry


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="LLM Customization Ops")
    app.include_router(router)
    init_telemetry(app, "llm-customization-ops", settings.otel_endpoint)

    loaded = load_model(settings.training.base_model, fake=settings.fake_model)
    app.state.loaded_model = loaded
    app.state.registry_path = (
        settings.paths.repo_root / "config" / "prompt_templates.json"
    )
    app.state.base_model = settings.training.base_model
    return app


app = create_app()
