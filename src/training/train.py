"""
Step 4: Fine-tune Legal-BERT on CUAD with HuggingFace Trainer + W&B.

CPU quick mode (~20-30 min):  python -m src.training.train --quick
Full GPU training  (~2 hrs):  python -m src.training.train

Logs to W&B: training loss, position accuracy per step, and full EM/F1 post-training.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.data.cuad_dataset import CUADDataset, load_squad_json
from src.training.baseline import predict_sample
from src.training.squad_metrics import best_em_f1, compute_metrics

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data" / "raw"
MODEL_DIR  = ROOT / "models"
MODEL_NAME = os.getenv("BASE_MODEL", "nlpaueb/legal-bert-base-uncased")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="300 train / 60 eval samples, 3 epochs — CPU-friendly (~25 min)")
    p.add_argument("--train_samples", type=int, default=None, help="Override sample count")
    p.add_argument("--eval_samples",  type=int, default=None)
    p.add_argument("--epochs",        type=int, default=None)
    p.add_argument("--batch_size",    type=int, default=None)
    p.add_argument("--run_name",      type=str, default=None)
    return p.parse_args()


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_position_metrics(eval_pred):
    """
    Token-position accuracy — fast proxy metric computed during each eval step.
    Separate from the full EM/F1 which requires decoding back to text.
    """
    preds, labels = eval_pred.predictions, eval_pred.label_ids

    start_logits, end_logits = preds[0], preds[1]
    start_labels, end_labels = labels[0], labels[1]

    start_preds = start_logits.argmax(-1)
    end_preds   = end_logits.argmax(-1)

    start_acc = (start_preds == start_labels).mean()
    end_acc   = (end_preds   == end_labels).mean()
    span_acc  = ((start_preds == start_labels) & (end_preds == end_labels)).mean()

    return {
        "start_acc": round(float(start_acc), 4),
        "end_acc":   round(float(end_acc),   4),
        "span_acc":  round(float(span_acc),  4),
    }


# ── post-training EM/F1 eval ──────────────────────────────────────────────────

def run_squad_eval(samples, tokenizer, model, device, label="eval", n=None):
    """
    Decode predicted spans back to text and compute SQuAD EM + F1.
    This is the 'real' metric — done once after training, not per epoch.
    """
    if n:
        samples = samples[:n]
    predictions = {}
    references  = {}
    for s in samples:
        pred = predict_sample(s, tokenizer, model, device)
        predictions[s["id"]] = pred
        references[s["id"]]  = [a["text"] for a in s["answers"]]

    metrics = compute_metrics(predictions, references)
    print(f"\n  [{label}] Exact Match: {metrics['exact_match']:.2f}%  "
          f"F1: {metrics['f1']:.2f}%  (n={metrics['n_samples']})")
    return metrics


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── config ────────────────────────────────────────────────────────────────
    on_gpu = DEVICE == "cuda"

    if args.quick or not on_gpu:
        # CPU-practical config
        # 50 samples x 5 windows = 250 windows / batch 4 = 62 steps/epoch x 3 = 186 steps
        # ~2s/step at max_len=256 => ~6 min total. W&B sees data at step 1.
        train_n              = args.train_samples or 50
        eval_n               = args.eval_samples  or 20
        num_epochs           = args.epochs        or 3
        batch_size           = args.batch_size    or 4
        grad_accum           = 1      # 1 step = 1 update — W&B sees loss immediately
        max_len              = 256    # 4x faster than 512 (attention is O(n^2))
        stride               = 64
        max_windows          = 5      # cap windows per sample — the key CPU fix
        fp16                 = False
        run_label            = "quick-cpu"
    else:
        train_n              = args.train_samples or len(load_squad_json(DATA_DIR / "cuad_train.json"))
        eval_n               = args.eval_samples  or 500
        num_epochs           = args.epochs        or 5
        batch_size           = args.batch_size    or 16
        grad_accum           = 2
        max_len              = 512
        stride               = 128
        max_windows          = None   # use all windows for full GPU training
        fp16                 = True
        run_label            = "full-gpu"

    run_name = args.run_name or f"step4-finetune-{run_label}"

    print(f"Device     : {DEVICE}")
    print(f"Mode       : {run_label}")
    print(f"Train N    : {train_n}")
    print(f"Eval N     : {eval_n}")
    print(f"Epochs     : {num_epochs}")
    print(f"Batch size : {batch_size} (x{grad_accum} grad accum = {batch_size*grad_accum} effective)")
    print(f"Max length : {max_len} tokens | Stride: {stride} | Max windows/sample: {max_windows}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "contract-intelligence")
    os.environ["WANDB_ENTITY"]  = os.getenv("WANDB_ENTITY",  "chennamdilip1-bur")

    # ── load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_samples = load_squad_json(DATA_DIR / "cuad_train.json")[:train_n]
    eval_samples  = load_squad_json(DATA_DIR / "cuad_test.json")[:eval_n]
    print(f"  Train: {len(train_samples)} samples | Eval: {len(eval_samples)} samples")

    # ── tokenizer + model ─────────────────────────────────────────────────────
    tok_dir = MODEL_DIR / "tokenizer"
    if tok_dir.exists():
        print(f"Loading tokenizer from {tok_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    else:
        print(f"Loading tokenizer from HuggingFace: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.1f}M parameters")

    # ── datasets ──────────────────────────────────────────────────────────────
    print(f"\nTokenizing with sliding window (max_len={max_len}, stride={stride}, max_win={max_windows})...")
    train_ds = CUADDataset(train_samples, tokenizer, max_length=max_len, stride=stride,
                           split="train", max_windows_per_sample=max_windows)
    eval_ds  = CUADDataset(eval_samples,  tokenizer, max_length=max_len, stride=stride,
                           split="test",  max_windows_per_sample=max_windows)
    print(f"  Train: {len(train_ds)} windows | Eval: {len(eval_ds)} windows")

    # ── training args ─────────────────────────────────────────────────────────
    output_dir = str(MODEL_DIR / "checkpoints" / run_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,

        # Core schedule
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",

        # Eval + saving
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="span_acc",
        greater_is_better=True,
        save_total_limit=2,

        # Speed
        fp16=fp16 and on_gpu,
        dataloader_num_workers=0,

        # Logging — every step so W&B gets data immediately
        logging_steps=1,
        logging_first_step=True,
        report_to="wandb",

        # Reproducibility
        seed=42,
        data_seed=42,
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_position_metrics,
        processing_class=tokenizer,
    )

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting fine-tuning (run: {run_name})...")
    print(f"  Steps per epoch : {len(train_ds) // (batch_size * grad_accum)}")
    print(f"  Total steps     : {len(train_ds) // (batch_size * grad_accum) * num_epochs}")

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)

    # ── save best model ───────────────────────────────────────────────────────
    final_dir = str(MODEL_DIR / "final" / run_name)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    # ── full EM/F1 evaluation ─────────────────────────────────────────────────
    print("\nRunning full SQuAD EM/F1 evaluation (decode spans to text)...")
    model.eval()

    train_metrics = run_squad_eval(train_samples, tokenizer, model, DEVICE,
                                   label="train", n=min(50, train_n))
    eval_metrics  = run_squad_eval(eval_samples,  tokenizer, model, DEVICE,
                                   label="eval",  n=min(50, eval_n))

    wandb.log({
        "final/train_em": train_metrics["exact_match"],
        "final/train_f1": train_metrics["f1"],
        "final/eval_em":  eval_metrics["exact_match"],
        "final/eval_f1":  eval_metrics["f1"],
    })

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("FINE-TUNING SUMMARY")
    print("=" * 55)
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Train N   : {train_n} samples | {len(train_ds)} windows")
    print(f"  Epochs    : {num_epochs}")
    print(f"  Train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"\n  SQuAD metrics (50-sample post-eval):")
    print(f"    Train  EM={train_metrics['exact_match']:.2f}%  F1={train_metrics['f1']:.2f}%")
    print(f"    Eval   EM={eval_metrics['exact_match']:.2f}%   F1={eval_metrics['f1']:.2f}%")
    print(f"\n  Baseline (zero-shot): EM=0.00%  F1=8.31%")
    print(f"  Improvement: +{eval_metrics['exact_match']:.2f}pp EM  +{eval_metrics['f1']-8.31:.2f}pp F1")
    print(f"\n  Saved: {final_dir}")
    print("=" * 55)
    print("\nStep 4 complete. Ready for Step 5: per-clause-type evaluation.")


if __name__ == "__main__":
    main()
