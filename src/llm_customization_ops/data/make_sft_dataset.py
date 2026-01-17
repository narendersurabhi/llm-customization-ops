from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from llm_customization_ops.data.schema import SFTRecord, validate_jsonl
from llm_customization_ops.ops.utils import ensure_dir


def load_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_sft_dataset(input_path: Path, output_dir: Path) -> Dataset:
    records = load_jsonl(input_path)
    validated = validate_jsonl(records, SFTRecord)
    dataset = Dataset.from_list([item.model_dump() for item in validated])
    ensure_dir(output_dir)
    dataset.save_to_disk(str(output_dir))
    return dataset
