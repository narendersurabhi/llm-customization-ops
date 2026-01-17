from pathlib import Path

from llm_customization_ops.data.make_pref_dataset import build_pref_dataset
from llm_customization_ops.data.make_sft_dataset import build_sft_dataset


def test_build_sft_dataset(tmp_path: Path) -> None:
    out_dir = tmp_path / "sft"
    dataset = build_sft_dataset(
        Path("src/llm_customization_ops/data/fixtures/sft.jsonl"), out_dir
    )
    assert dataset.num_rows == 3


def test_build_pref_dataset(tmp_path: Path) -> None:
    out_dir = tmp_path / "pref"
    dataset = build_pref_dataset(
        Path("src/llm_customization_ops/data/fixtures/pref.jsonl"),
        out_dir,
    )
    assert dataset.num_rows == 2
