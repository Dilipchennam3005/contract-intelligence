"""
Step 2: CUAD Data Exploration
Analyzes clause-type distribution, contract length, answer span stats.
Logs all charts + tables to W&B.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR  = Path(__file__).resolve().parents[2] / "data" / "raw"
FIGS_DIR  = Path(__file__).resolve().parents[2] / "data" / "processed"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_squad_json(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                samples.append({
                    "id":      qa["id"],
                    "title":   article.get("title", ""),
                    "context": ctx,
                    "question": qa["question"],
                    "answers": qa["answers"],
                })
    return samples


def extract_clause_name(question: str) -> str:
    if '"' in question:
        return question.split('"')[1]
    return question[:50]


# ── analysis ──────────────────────────────────────────────────────────────────

def clause_distribution(samples):
    stats = defaultdict(lambda: {"total": 0, "positive": 0})
    for s in samples:
        name = extract_clause_name(s["question"])
        stats[name]["total"] += 1
        if s["answers"]:
            stats[name]["positive"] += 1
    rows = []
    for clause, st in stats.items():
        rate = st["positive"] / st["total"] * 100
        rows.append({"clause": clause, "positive_pct": round(rate, 1),
                     "total": st["total"], "positive": st["positive"]})
    rows.sort(key=lambda r: r["positive_pct"], reverse=True)
    return rows


def context_length_stats(samples):
    return [len(s["context"].split()) for s in samples]


def answer_span_stats(samples):
    spans = []
    for s in samples:
        for ans in s["answers"]:
            spans.append(len(ans["text"].split()))
    return spans


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_clause_distribution(rows, split_name):
    clauses = [r["clause"] for r in rows]
    rates   = [r["positive_pct"] for r in rows]
    colors  = ["#e74c3c" if r < 20 else "#f39c12" if r < 50 else "#27ae60" for r in rates]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(clauses[::-1], rates[::-1], color=colors[::-1])
    ax.set_xlabel("% of contracts containing clause", fontsize=12)
    ax.set_title(f"CUAD Clause-Type Distribution ({split_name})", fontsize=14)
    ax.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="20% threshold")
    ax.legend()
    plt.tight_layout()
    path = FIGS_DIR / f"clause_distribution_{split_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_context_lengths(lengths, split_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(x=380, color="red", linestyle="--", label="~512-token BERT limit (~380 words)")
    ax.set_xlabel("Context length (words)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Contract Context Length Distribution ({split_name})", fontsize=14)
    ax.legend()
    plt.tight_layout()
    path = FIGS_DIR / f"context_lengths_{split_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_answer_spans(spans, split_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    clipped = [min(s, 100) for s in spans]
    ax.hist(clipped, bins=40, color="#2ecc71", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Answer span length (words, capped at 100)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Answer Span Lengths ({split_name})", fontsize=14)
    plt.tight_layout()
    path = FIGS_DIR / f"answer_spans_{split_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "contract-intelligence"),
        entity=os.getenv("WANDB_ENTITY", "chennamdilip1-bur"),
        job_type="data-exploration",
        name="step2-eda",
    )

    for split in ["train", "test"]:
        path = DATA_DIR / f"cuad_{split}.json"
        print(f"\nLoading {split}...")
        samples = load_squad_json(path)

        n_contracts  = len({s["title"] for s in samples})
        rows         = clause_distribution(samples)
        ctx_lengths  = context_length_stats(samples)
        span_lengths = answer_span_stats(samples)
        n_positive   = sum(1 for s in samples if s["answers"])

        print(f"  {len(samples):,} QA pairs | {n_contracts} unique contracts")
        print(f"  Context length : mean={np.mean(ctx_lengths):.0f}w  "
              f"median={np.median(ctx_lengths):.0f}w  max={max(ctx_lengths)}w")
        pct_pos = n_positive / len(samples) * 100
        print(f"  Positive (has answer): {n_positive:,}/{len(samples):,} = {pct_pos:.1f}%")
        print(f"  Answer span    : mean={np.mean(span_lengths):.1f}w  "
              f"median={np.median(span_lengths):.0f}w  max={max(span_lengths)}w")

        print(f"\n  Clause-type breakdown ({split}):")
        for r in rows:
            bar = "#" * int(r["positive_pct"] / 5)
            print(f"    {r['positive_pct']:5.1f}%  {bar:<20}  {r['clause']}")

        p1 = plot_clause_distribution(rows, split)
        p2 = plot_context_lengths(ctx_lengths, split)
        p3 = plot_answer_spans(span_lengths, split)
        print(f"\n  Charts saved to {FIGS_DIR}")

        prefix = f"{split}/"
        wandb.log({
            f"{prefix}num_qa_pairs":      len(samples),
            f"{prefix}num_contracts":     n_contracts,
            f"{prefix}ctx_len_mean":      round(float(np.mean(ctx_lengths)), 1),
            f"{prefix}ctx_len_median":    int(np.median(ctx_lengths)),
            f"{prefix}ctx_len_max":       max(ctx_lengths),
            f"{prefix}pct_positive":      round(pct_pos, 2),
            f"{prefix}span_len_mean":     round(float(np.mean(span_lengths)), 1),
            f"{prefix}clause_dist_chart": wandb.Image(p1),
            f"{prefix}ctx_len_chart":     wandb.Image(p2),
            f"{prefix}span_len_chart":    wandb.Image(p3),
        })

        table = wandb.Table(columns=["clause", "positive_pct", "positive", "total"])
        for r in rows:
            table.add_data(r["clause"], r["positive_pct"], r["positive"], r["total"])
        wandb.log({f"{prefix}clause_table": table})

    print("\n" + "=" * 60)
    print("KEY DESIGN DECISIONS FOR STEP 3+")
    print("=" * 60)
    print("""
1. TASK FRAMING: Span-extraction (SQuAD-style), not binary classification.
   Reason: gives richer output (the actual clause text, not just yes/no).

2. CONTEXT LENGTH: Many contracts exceed 512 tokens (BERT limit ~380 words).
   Solution: Sliding window tokenization with stride=128.

3. CLASS IMBALANCE: 18/41 clauses appear in <30% of contracts.
   Solution: Evaluate with macro-F1. Tune decision threshold per clause at inference.

4. METRICS: Exact Match + F1 on answer spans (SQuAD standard),
   plus per-clause-type F1 logged to W&B for visibility.
""")

    run.finish()
    print("Step 2 complete. Charts in data/processed/ and logged to W&B.")


if __name__ == "__main__":
    main()
