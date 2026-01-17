from __future__ import annotations

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from llm_customization_ops.ops.tracing import configure_tracing


def init_telemetry(app, service_name: str, endpoint: str | None) -> None:
    configure_tracing(service_name, endpoint)
    FastAPIInstrumentor.instrument_app(app)
