from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_customization_ops.eval.metrics import accuracy, exact_match, rouge_like
from llm_customization_ops.ops.utils import ensure_dir, write_json


def run_eval(golden_path: Path, output_dir: Path) -> dict[str, Any]:
    ensure_dir(output_dir)
    records = [
        json.loads(line)
        for line in golden_path.read_text().splitlines()
        if line.strip()
    ]
    summaries = []
    classifications = []
    extractions = []

    for row in records:
        if row["task"] == "summarization":
            summaries.append((row["prediction"], row["target"]))
        elif row["task"] == "classification":
            classifications.append((row["prediction"], row["target"]))
        elif row["task"] == "extraction":
            extractions.append((row["prediction"], row["target"]))

    summary_scores = [rouge_like(p, t) for p, t in summaries]
    summary_score = sum(summary_scores) / max(1, len(summary_scores))
    class_score = accuracy(
        [p for p, _ in classifications], [t for _, t in classifications]
    )
    extract_scores = [exact_match(p, t) for p, t in extractions]
    extract_score = sum(extract_scores) / max(1, len(extract_scores))

    report = {
        "summarization": {"rouge_like": summary_score},
        "classification": {"accuracy": class_score},
        "extraction": {"exact_match": extract_score},
    }
    write_json(output_dir / "report.json", report)
    return report
