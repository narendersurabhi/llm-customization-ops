from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_thresholds(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def gate_report(report_path: Path, thresholds_path: Path) -> list[str]:
    report = json.loads(report_path.read_text())
    thresholds = load_thresholds(thresholds_path)
    failures: list[str] = []
    for metric_group, metric_thresholds in thresholds.items():
        for metric, threshold in metric_thresholds.items():
            value = report.get(metric_group, {}).get(metric, 0)
            if value < threshold:
                failures.append(
                    f"{metric_group}.{metric}={value:.3f} below threshold {threshold:.3f}"
                )
    return failures
