"""
Post-training evaluation: compare fine-tuned model EM/F1 against zero-shot baseline.
Run: python -m src.training.eval_finetuned
"""
import os
from collections import defaultdict
from pathlib import Path

import wandb
from dotenv import load_dotenv
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.data.cuad_dataset import load_squad_json
from src.training.baseline import predict_sample
from src.training.squad_metrics import best_em_f1, compute_metrics

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "final" / "step4-finetune-quick-cpu"
DATA_DIR  = ROOT / "data" / "raw"
N_EVAL    = 50
DEVICE    = "cpu"
BASELINE_F1 = 8.31


def main():
    print(f"Loading fine-tuned model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForQuestionAnswering.from_pretrained(str(MODEL_DIR))
    model.eval()

    print(f"Running EM/F1 on {N_EVAL} test samples...\n")
    test_samples = load_squad_json(DATA_DIR / "cuad_test.json")[:N_EVAL]

    predictions = {}
    references  = {}
    clause_f1   = defaultdict(list)

    for i, s in enumerate(test_samples):
        pred = predict_sample(s, tokenizer, model, DEVICE)
        sid  = s["id"]
        gts  = [a["text"] for a in s["answers"]]

        predictions[sid] = pred
        references[sid]  = gts

        clause = sid.split("__")[-1] if "__" in sid else "unknown"
        _, f1  = best_em_f1(pred, gts)
        clause_f1[clause].append(f1)

        if i < 5:
            print(f"  [{i+1}] clause: {clause}")
            print(f"        gold: {gts[0][:80] if gts else '(none)'}")
            print(f"        pred: {pred[:80] if pred else '(none)'}")
            print(f"        F1  : {f1:.2f}\n")

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{N_EVAL} done")

    metrics = compute_metrics(predictions, references)

    print("\n" + "=" * 50)
    print("FINAL RESULTS (fine-tuned Legal-BERT)")
    print("=" * 50)
    print(f"  Exact Match : {metrics['exact_match']:.2f}%  (baseline: 0.00%)")
    print(f"  F1          : {metrics['f1']:.2f}%  (baseline: {BASELINE_F1}%)")
    print(f"  Improvement : +{metrics['exact_match']:.2f}pp EM  "
          f"+{metrics['f1'] - BASELINE_F1:.2f}pp F1")
    print("=" * 50)

    print("\nPer-clause F1 (fine-tuned):")
    for clause, scores in sorted(clause_f1.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(scores) / len(scores) * 100
        print(f"  {avg:5.1f}%  {clause}")

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "contract-intelligence"),
        entity=os.getenv("WANDB_ENTITY", "chennamdilip1-bur"),
        job_type="eval",
        name="step4-final-eval",
    )
    wandb.log({
        "final/eval_em":          metrics["exact_match"],
        "final/eval_f1":          metrics["f1"],
        "final/baseline_f1":      BASELINE_F1,
        "final/improvement_f1":   round(metrics["f1"] - BASELINE_F1, 2),
    })

    table = wandb.Table(columns=["clause", "f1_pct"])
    for clause, scores in clause_f1.items():
        table.add_data(clause, round(sum(scores) / len(scores) * 100, 2))
    wandb.log({"final/per_clause_f1": table})

    run.finish()
    print("\nStep 4 evaluation complete. Results logged to W&B.")


if __name__ == "__main__":
    main()
