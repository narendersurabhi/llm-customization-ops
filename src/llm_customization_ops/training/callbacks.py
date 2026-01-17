from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from transformers import TrainerCallback


@dataclass
class RunSummary:
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class SummaryCallback(TrainerCallback):
    def __init__(self, summary: RunSummary) -> None:
        self.summary = summary

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        self.summary.end_time = time.time()
        self.summary.metrics = dict(state.log_history[-1]) if state.log_history else {}
        return control
