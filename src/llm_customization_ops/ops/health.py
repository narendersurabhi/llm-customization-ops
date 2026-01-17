from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HealthStatus:
    ok: bool
    message: str


def ready_status(model_loaded: bool) -> HealthStatus:
    if model_loaded:
        return HealthStatus(ok=True, message="ready")
    return HealthStatus(ok=False, message="model_not_loaded")
