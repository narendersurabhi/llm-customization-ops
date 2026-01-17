from pathlib import Path

from llm_customization_ops.eval.gates import gate_report


def test_gate_report_passes(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        '{"summarization": {"rouge_like": 0.9}, '
        '"classification": {"accuracy": 0.9}, '
        '"extraction": {"exact_match": 1.0}}'
    )
    thresholds = Path("config/eval_gates.json")
    failures = gate_report(report_path, thresholds)
    assert failures == []
