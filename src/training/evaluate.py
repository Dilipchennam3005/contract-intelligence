"""
Step 5: Proper per-clause evaluation of the fine-tuned Legal-BERT model.

Breaks down performance into:
  - Overall EM / F1
  - Has-answer F1  (gold answer exists — did we find the right span?)
  - No-answer accuracy (no clause present — did we correctly say nothing?)
  - Per-clause precision / recall / F1
  - Confusion matrix: TP / FP / FN / TN per clause type

Run: python -m src.training.evaluate
"""
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.data.cuad_dataset import load_squad_json
from src.training.squad_metrics import best_em_f1, compute_metrics

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "final" / "step4-finetune-quick-cpu"
DATA_DIR  = ROOT / "data" / "raw"
FIGS_DIR  = ROOT / "data" / "processed"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

N_EVAL    = 200   # test samples to evaluate
MAX_LEN   = 256   # match training config for speed
STRIDE    = 64
MAX_WIN   = 10    # more windows than training (5) for better coverage
DEVICE    = "cpu"
BASELINE_F1 = 8.31


# ── fast predict (respects max_windows for speed) ────────────────────────────

def predict(sample: dict, tokenizer, model) -> tuple[str, float]:
    """Returns (predicted_text, best_score). Empty string = no answer predicted."""
    encoding = tokenizer(
        sample["question"],
        sample["context"],
        max_length=MAX_LEN,
        stride=STRIDE,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoding.pop("offset_mapping")
    encoding.pop("overflow_to_sample_mapping", None)

    n_windows = min(encoding["input_ids"].shape[0], MAX_WIN)
    best_score, best_text = float("-inf"), ""

    model.eval()
    with torch.no_grad():
        for i in range(n_windows):
            win = {k: v[i].unsqueeze(0) for k, v in encoding.items()}
            out = model(**win)
            s = out.start_logits[0].cpu()
            e = out.end_logits[0].cpu()

            start = int(s.argmax())
            end   = int(e.argmax())

            if start == 0 or end < start:
                score = (s[0] + e[0]).item()
                if score > best_score:
                    best_score, best_text = score, ""
                continue

            score = (s[start] + e[end]).item()
            if score > best_score:
                ids = encoding["input_ids"][i][start: end + 1]
                best_text  = tokenizer.decode(ids, skip_special_tokens=True).strip()
                best_score = score

    return best_text, best_score


# ── per-clause precision / recall / F1 ───────────────────────────────────────

def clause_prf(records: list[dict], f1_threshold: float = 0.3) -> dict:
    """
    For each clause type compute P / R / F1 treating clause presence as binary.
    A prediction counts as 'found' if span F1 >= f1_threshold (or both empty).
    """
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for r in records:
        clause = r["clause"]
        gold_positive = bool(r["gts"])
        pred_positive = bool(r["pred"])

        if gold_positive and pred_positive:
            if r["f1"] >= f1_threshold:
                stats[clause]["tp"] += 1
            else:
                stats[clause]["fp"] += 1
                stats[clause]["fn"] += 1
        elif not gold_positive and not pred_positive:
            stats[clause]["tn"] += 1
        elif gold_positive and not pred_positive:
            stats[clause]["fn"] += 1
        else:  # pred_positive but gold empty
            stats[clause]["fp"] += 1

    results = {}
    for clause, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        results[clause] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": s["tn"],
            "support": tp + fn,
        }
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_clause_f1(clause_results: dict) -> str:
    rows = sorted(clause_results.items(), key=lambda x: -x[1]["f1"])
    clauses = [r[0] for r in rows]
    f1s     = [r[1]["f1"] * 100 for r in rows]
    precs   = [r[1]["precision"] * 100 for r in rows]
    recs    = [r[1]["recall"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(13, 9))
    x = np.arange(len(clauses))
    w = 0.28
    ax.barh(x + w,   f1s[::-1],   w, label="F1",        color="#3498db")
    ax.barh(x,       precs[::-1], w, label="Precision",  color="#2ecc71")
    ax.barh(x - w,   recs[::-1],  w, label="Recall",     color="#e74c3c")
    ax.set_yticks(x)
    ax.set_yticklabels(clauses[::-1], fontsize=8)
    ax.set_xlabel("Score (%)")
    ax.set_title("Per-Clause Precision / Recall / F1 (fine-tuned Legal-BERT)", fontsize=13)
    ax.legend()
    ax.axvline(50, color="gray", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = str(FIGS_DIR / "step5_clause_prf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_has_vs_no_answer(has_ans_f1: float, no_ans_acc: float,
                           baseline_f1: float) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Has-answer F1
    axes[0].bar(["Baseline\n(zero-shot)", "Fine-tuned"],
                [baseline_f1, has_ans_f1],
                color=["#e74c3c", "#27ae60"], width=0.4)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("F1 (%)")
    axes[0].set_title("Has-Answer F1\n(clause IS present in contract)")
    for i, v in enumerate([baseline_f1, has_ans_f1]):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    # No-answer accuracy
    axes[1].bar(["No-Answer\nAccuracy"], [no_ans_acc],
                color="#3498db", width=0.3)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("No-Answer Accuracy\n(clause NOT in contract)")
    axes[1].text(0, no_ans_acc + 1, f"{no_ans_acc:.1f}%", ha="center", fontweight="bold")

    plt.suptitle("Fine-tuned Legal-BERT — Split Evaluation", fontsize=13)
    plt.tight_layout()
    path = str(FIGS_DIR / "step5_split_eval.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "contract-intelligence"),
        entity=os.getenv("WANDB_ENTITY",   "chennamdilip1-bur"),
        job_type="evaluation",
        name="step5-proper-eval",
        config={"n_eval": N_EVAL, "max_len": MAX_LEN, "max_windows": MAX_WIN},
    )

    print(f"Loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model     = AutoModelForQuestionAnswering.from_pretrained(str(MODEL_DIR))
    model.eval()

    print(f"\nEvaluating on {N_EVAL} test samples (max {MAX_WIN} windows each)...\n")
    samples = load_squad_json(DATA_DIR / "cuad_test.json")[:N_EVAL]

    records = []
    predictions, references = {}, {}

    for i, s in enumerate(samples):
        pred_text, _ = predict(s, tokenizer, model)
        gts = [a["text"] for a in s["answers"]]
        clause = s["id"].split("__")[-1] if "__" in s["id"] else "unknown"
        em, f1 = best_em_f1(pred_text, gts)

        records.append({
            "id": s["id"], "clause": clause,
            "pred": pred_text, "gts": gts,
            "em": em, "f1": f1,
            "gold_positive": bool(gts),
            "pred_positive": bool(pred_text),
        })
        predictions[s["id"]] = pred_text
        references[s["id"]]  = gts

        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{N_EVAL} done")

    # ── overall metrics ────────────────────────────────────────────────────
    overall = compute_metrics(predictions, references)

    has_ans = [r for r in records if r["gold_positive"]]
    no_ans  = [r for r in records if not r["gold_positive"]]

    has_ans_em  = np.mean([r["em"] for r in has_ans]) * 100 if has_ans else 0
    has_ans_f1  = np.mean([r["f1"] for r in has_ans]) * 100 if has_ans else 0
    no_ans_acc  = np.mean([not r["pred_positive"] for r in no_ans]) * 100 if no_ans else 0
    false_pos_n = sum(1 for r in no_ans if r["pred_positive"])
    false_neg_n = sum(1 for r in has_ans if not r["pred_positive"])

    # ── per-clause PRF ─────────────────────────────────────────────────────
    clause_results = clause_prf(records)

    # ── print report ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — EVALUATION REPORT")
    print("=" * 60)

    print(f"\n[OVERALL]  n={N_EVAL}")
    print(f"  Exact Match : {overall['exact_match']:.2f}%")
    print(f"  F1          : {overall['f1']:.2f}%")

    print(f"\n[HAS-ANSWER subset]  n={len(has_ans)} samples where clause IS present")
    print(f"  Exact Match : {has_ans_em:.2f}%")
    print(f"  F1          : {has_ans_f1:.2f}%")
    print(f"  False Negatives (missed clauses): {false_neg_n}")

    print(f"\n[NO-ANSWER subset]   n={len(no_ans)} samples where clause NOT present")
    print(f"  Accuracy    : {no_ans_acc:.2f}%  (correctly said 'no clause')")
    print(f"  False Positives (hallucinated clauses): {false_pos_n}")

    print(f"\n[PER-CLAUSE F1]")
    print(f"  {'Clause':<40} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for clause, m in sorted(clause_results.items(), key=lambda x: -x[1]["f1"]):
        print(f"  {clause:<40} {m['precision']*100:>5.1f}% {m['recall']*100:>5.1f}% "
              f"{m['f1']*100:>5.1f}%  {m['support']:>6}")

    macro_f1 = np.mean([m["f1"] for m in clause_results.values()]) * 100
    print(f"\n  Macro-F1 across {len(clause_results)} clause types: {macro_f1:.2f}%")

    # ── plots ──────────────────────────────────────────────────────────────
    p1 = plot_clause_f1(clause_results)
    p2 = plot_has_vs_no_answer(has_ans_f1, no_ans_acc, BASELINE_F1)
    print(f"\nCharts saved to {FIGS_DIR}")

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb.log({
        "eval/overall_em":       overall["exact_match"],
        "eval/overall_f1":       overall["f1"],
        "eval/has_answer_em":    round(has_ans_em, 2),
        "eval/has_answer_f1":    round(has_ans_f1, 2),
        "eval/no_answer_acc":    round(no_ans_acc, 2),
        "eval/false_positives":  false_pos_n,
        "eval/false_negatives":  false_neg_n,
        "eval/macro_f1":         round(macro_f1, 2),
        "eval/n_has_answer":     len(has_ans),
        "eval/n_no_answer":      len(no_ans),
        "eval/clause_prf_chart": wandb.Image(p1),
        "eval/split_eval_chart": wandb.Image(p2),
    })

    prf_table = wandb.Table(
        columns=["clause", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "support"])
    for clause, m in clause_results.items():
        prf_table.add_data(clause, m["precision"], m["recall"], m["f1"],
                           m["tp"], m["fp"], m["fn"], m["tn"], m["support"])
    wandb.log({"eval/per_clause_prf": prf_table})

    run.finish()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print(f"""
  1. Overall F1 is inflated by no-answer majority:
       Overall F1     = {overall['f1']:.1f}%
       Has-answer F1  = {has_ans_f1:.1f}%  <- the honest number
       No-answer acc  = {no_ans_acc:.1f}%

  2. False negatives ({false_neg_n}) >> False positives ({false_pos_n})
     The model errs toward saying "nothing" — safe for a legal tool but
     means it misses clauses. Fix: train with more windows (GPU + full data).

  3. Macro-F1 = {macro_f1:.1f}% — penalises ignoring rare clauses.

  Next: Step 7 — FastAPI service wrapping this model.
""")


if __name__ == "__main__":
    main()
