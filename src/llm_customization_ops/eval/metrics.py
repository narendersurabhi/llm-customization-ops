from __future__ import annotations

from collections.abc import Iterable


def exact_match(pred: str, target: str) -> float:
    return float(pred.strip() == target.strip())


def accuracy(preds: Iterable[str], targets: Iterable[str]) -> float:
    pred_list = list(preds)
    target_list = list(targets)
    correct = sum(p == t for p, t in zip(pred_list, target_list, strict=False))
    return correct / max(1, len(target_list))


def rouge_like(pred: str, target: str) -> float:
    pred_tokens = pred.lower().split()
    target_tokens = target.lower().split()
    overlap = len(set(pred_tokens) & set(target_tokens))
    return overlap / max(1, len(set(target_tokens)))
