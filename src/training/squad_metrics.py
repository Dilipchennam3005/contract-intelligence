"""
SQuAD-style Exact Match and F1 for span extraction.
Ported from the official SQuAD evaluation script.
"""
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens  = normalize_answer(prediction).split()
    gold_tokens  = normalize_answer(ground_truth).split()
    common       = Counter(pred_tokens) & Counter(gold_tokens)
    num_same     = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_em_f1(prediction: str, ground_truths: list[str]) -> tuple[float, float]:
    """Return best EM and F1 across all ground truth answers."""
    if not ground_truths:
        return (1.0 if prediction == "" else 0.0,
                1.0 if prediction == "" else 0.0)
    em = max(exact_match(prediction, gt) for gt in ground_truths)
    f1 = max(f1_score(prediction, gt)    for gt in ground_truths)
    return em, f1


def compute_metrics(predictions: dict[str, str], references: dict[str, list[str]]) -> dict:
    """
    predictions: {sample_id: predicted_text}
    references:  {sample_id: [answer_text, ...]}
    """
    total_em, total_f1 = 0.0, 0.0
    n = len(predictions)
    for sid, pred in predictions.items():
        gts = references.get(sid, [])
        em, f1 = best_em_f1(pred, gts)
        total_em += em
        total_f1 += f1
    return {
        "exact_match": round(total_em / n * 100, 2) if n else 0.0,
        "f1":          round(total_f1 / n * 100, 2) if n else 0.0,
        "n_samples":   n,
    }
