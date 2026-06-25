"""
Step 3: Baseline Legal-BERT QA — zero-shot inference (no fine-tuning).

Confirms the full pipeline works end-to-end:
  load data -> tokenize (sliding window) -> model forward pass -> decode span -> evaluate

Run: python src/training/baseline.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.data.cuad_dataset import CUADDataset, load_squad_json
from src.training.squad_metrics import best_em_f1, compute_metrics

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "raw"
MODEL_DIR   = ROOT / "models"
MODEL_NAME  = os.getenv("BASE_MODEL", "nlpaueb/legal-bert-base-uncased")
MAX_LEN     = 512
STRIDE      = 128
N_BASELINE  = 100   # number of raw samples for quick sanity check
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── span decoding ──────────────────────────────────────────────────────────────

def decode_span(
    input_ids: torch.Tensor,
    start_logits: torch.Tensor,
    end_logits:   torch.Tensor,
    tokenizer,
    max_answer_len: int = 100,
) -> str:
    """
    Greedy best valid span: pick argmax start, then find best end >= start
    within max_answer_len. Returns empty string if start == 0 (CLS = no answer).
    """
    start_idx = int(start_logits.argmax())
    if start_idx == 0:
        return ""
    # restrict end search to [start, start + max_answer_len)
    end_logits_masked = end_logits.clone()
    end_logits_masked[:start_idx] = float("-inf")
    end_logits_masked[start_idx + max_answer_len:] = float("-inf")
    end_idx = int(end_logits_masked.argmax())

    tokens = input_ids[start_idx : end_idx + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


# ── per-sample inference (handles multi-window merge) ─────────────────────────

def predict_sample(sample: dict, tokenizer, model, device: str) -> str:
    """
    Tokenize one (question, context) pair with sliding window,
    run each window through the model, return the span with highest
    combined start+end logit across all windows.
    """
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
    offset_mapping = encoding.pop("offset_mapping").tolist()
    encoding.pop("overflow_to_sample_mapping", None)

    best_score = float("-inf")
    best_text  = ""

    model.eval()
    with torch.no_grad():
        for win_idx in range(encoding["input_ids"].shape[0]):
            window_enc = {k: v[win_idx].unsqueeze(0).to(device) for k, v in encoding.items()}
            outputs = model(**window_enc)
            s_logits = outputs.start_logits[0].cpu()
            e_logits = outputs.end_logits[0].cpu()

            start_idx = int(s_logits.argmax())
            end_idx   = int(e_logits.argmax())

            if start_idx == 0 or end_idx < start_idx:
                score = s_logits[0].item() + e_logits[0].item()  # CLS score
                if score > best_score:
                    best_score = score
                    best_text  = ""
                continue

            score = s_logits[start_idx].item() + e_logits[end_idx].item()
            if score > best_score:
                best_score = score
                ids = encoding["input_ids"][win_idx][start_idx: end_idx + 1]
                best_text = tokenizer.decode(ids, skip_special_tokens=True).strip()

    return best_text


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading model: {MODEL_NAME}")

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "contract-intelligence"),
        entity=os.getenv("WANDB_ENTITY",   "chennamdilip1-bur"),
        job_type="baseline",
        name="step3-legal-bert-zeroshot",
        config={
            "model":      MODEL_NAME,
            "max_length": MAX_LEN,
            "stride":     STRIDE,
            "n_samples":  N_BASELINE,
            "device":     DEVICE,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")
    wandb.config.update({"n_params_M": round(n_params, 1)})

    # Load a small slice of train data for speed
    print(f"\nLoading {N_BASELINE} train samples...")
    all_samples = load_squad_json(DATA_DIR / "cuad_train.json")
    samples     = all_samples[:N_BASELINE]

    # ── verify dataset + dataloader ────────────────────────────────────────
    print("Building CUADDataset (sliding window tokenization)...")
    ds = CUADDataset(samples, tokenizer, max_length=MAX_LEN, stride=STRIDE, split="train")
    print(f"  {len(samples)} raw samples -> {len(ds)} windowed features")
    print(f"  Feature keys: {list(ds[0].keys())}")
    print(f"  Input IDs shape: {ds[0]['input_ids'].shape}")

    # Check that answer positions are reasonable
    has_answer = [(ds[i]["start_positions"].item(), ds[i]["end_positions"].item())
                  for i in range(len(ds))]
    n_answerable = sum(1 for s, e in has_answer if s > 0)
    print(f"  Windows with an answer: {n_answerable}/{len(ds)} "
          f"({n_answerable/len(ds)*100:.1f}%)")

    # ── zero-shot inference ────────────────────────────────────────────────
    print(f"\nRunning zero-shot inference on {N_BASELINE} samples...")
    predictions = {}
    references  = {}
    clause_results = defaultdict(list)  # clause_name -> list of F1 scores

    for i, sample in enumerate(samples):
        pred = predict_sample(sample, tokenizer, model, DEVICE)
        sid  = sample["id"]
        gts  = [a["text"] for a in sample["answers"]]
        clause = sample["id"].split("__")[-1] if "__" in sample["id"] else "unknown"

        predictions[sid] = pred
        references[sid]  = gts

        em, f1 = best_em_f1(pred, gts)
        clause_results[clause].append(f1)

        if i < 5:  # show first 5 predictions
            print(f"\n  [{i+1}] Clause: {clause}")
            print(f"        Question: {sample['question'][:80]}...")
            print(f"        Gold:     {gts[0][:80] if gts else '(no answer)'}")
            print(f"        Pred:     {pred[:80] if pred else '(no answer)'}")
            print(f"        EM={em:.0f}  F1={f1:.2f}")

        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{N_BASELINE} done")

    # ── overall metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(predictions, references)
    print(f"\nZero-shot baseline results (n={N_BASELINE}):")
    print(f"  Exact Match : {metrics['exact_match']:.2f}%")
    print(f"  F1          : {metrics['f1']:.2f}%")
    print(f"  (These are expected to be low — model has not been fine-tuned yet)")

    # Per-clause F1
    print("\nPer-clause F1 (zero-shot):")
    clause_f1 = {c: sum(v)/len(v)*100 for c, v in clause_results.items()}
    for clause, f1 in sorted(clause_f1.items(), key=lambda x: -x[1])[:15]:
        print(f"  {f1:5.1f}%  {clause}")

    # ── log to W&B ────────────────────────────────────────────────────────
    wandb.log({
        "baseline/exact_match":  metrics["exact_match"],
        "baseline/f1":           metrics["f1"],
        "baseline/n_windows":    len(ds),
        "baseline/n_answerable_windows": n_answerable,
    })

    table = wandb.Table(columns=["clause", "f1_pct"])
    for clause, f1 in sorted(clause_f1.items(), key=lambda x: -x[1]):
        table.add_data(clause, round(f1, 2))
    wandb.log({"baseline/per_clause_f1": table})

    # ── save tokenizer for re-use in fine-tuning ──────────────────────────
    tok_dir = MODEL_DIR / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tok_dir))
    print(f"\nTokenizer saved to {tok_dir}")

    run.finish()
    print("\nStep 3 complete. Zero-shot baseline established.")
    print("Next: Step 4 — Fine-tune with HuggingFace Trainer + W&B logging.")


if __name__ == "__main__":
    main()
