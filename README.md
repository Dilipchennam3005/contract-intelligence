# Contract Intelligence Pipeline

Fine-tuned Legal-BERT on the CUAD dataset for automated contract clause extraction. Deployed as a REST API with Docker.

## What it does

Takes contract text as input and identifies which of 41 legal clause types are present, extracting the relevant passage for each.

**Input:** Raw contract text  
**Output:** JSON with 41 clause types, extracted spans, and confidence scores

## Architecture

```
contract-intelligence/
├── src/
│   ├── data/
│   │   ├── download_cuad.py      # Download CUAD dataset from GitHub
│   │   ├── explore_cuad.py       # EDA + W&B charts
│   │   └── cuad_dataset.py       # PyTorch Dataset with sliding-window tokenization
│   └── training/
│       ├── baseline.py           # Zero-shot inference (Step 3)
│       ├── train.py              # HuggingFace Trainer fine-tuning (Step 4)
│       ├── evaluate.py           # Proper per-clause P/R/F1 (Step 5)
│       └── squad_metrics.py      # Exact Match + F1 scorer
├── api/
│   ├── main.py                   # FastAPI app (3 endpoints)
│   ├── model.py                  # Model singleton + inference logic
│   └── schemas.py                # Pydantic request/response schemas
├── Dockerfile                    # Multi-stage build
├── docker-compose.yml
├── requirements.txt              # Full training deps
└── requirements-api.txt          # Slim inference-only deps
```

## Model

| Property | Value |
|---|---|
| Base model | `nlpaueb/legal-bert-base-uncased` |
| Parameters | 108.9M |
| Task | Extractive QA (SQuAD-style span extraction) |
| Dataset | CUAD — 510 contracts, 41 clause types, 22,450 QA pairs |
| Tokenization | Sliding window: max_length=512, stride=128 |

## Dataset — CUAD

The [Contract Understanding Atticus Dataset](https://github.com/TheAtticusProject/cuad) contains 510 commercial legal contracts manually labelled for 41 clause categories.

| Split | QA pairs | Contracts |
|---|---|---|
| Train | 22,450 | 408 |
| Test | 4,182 | 102 |

Key challenge: **class imbalance**. Some clauses (Document Name, Parties) appear in 100% of contracts; others (Most Favored Nation, Price Restrictions) appear in under 8%.

## Training

### Step 3 — Zero-shot baseline (no fine-tuning)

```bash
python -m src.training.baseline
```

| Metric | Score |
|---|---|
| Exact Match | 0.00% |
| F1 | 8.31% |

The `qa_outputs` head is randomly initialised — this is the floor to beat.

### Step 4 — Fine-tuning

```bash
# Quick CPU run (~1 hr, 50 samples)
python -m src.training.train --quick

# Full GPU run (recommended, all 22k samples)
python -m src.training.train
```

W&B tracks training loss, span accuracy per epoch, and final EM/F1.

### Step 5 — Evaluation

```bash
python -m src.training.evaluate
```

The honest evaluation splits results into two subsets:

| Subset | What it measures |
|---|---|
| **Has-answer F1** | When a clause IS present — did we extract the right span? |
| **No-answer accuracy** | When a clause is NOT present — did we correctly say nothing? |

**Note on the quick-CPU model:** Trained on 50 samples with 5 windows each, the model learned the no-answer majority (98% of windows) and achieves 0% has-answer F1. Retraining on full data with a GPU resolves this — the pipeline code doesn't change, only the model weights.

## API

### Run with Docker (recommended)

```bash
docker compose up
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Run locally

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000
```

### Endpoints

#### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/models/final/step4-finetune-quick-cpu",
  "device": "cpu"
}
```

#### `GET /clauses`
Returns all 41 CUAD clause types the model checks for.

#### `POST /analyze`
```json
// Request
{
  "title": "Software License Agreement",
  "text": "THIS AGREEMENT is entered into as of January 1, 2024..."
}

// Response
{
  "contract_title": "Software License Agreement",
  "total_clauses_checked": 41,
  "clauses_found": 3,
  "results": [
    {
      "clause_type": "Governing Law",
      "found": true,
      "extracted_text": "governed by the laws of the State of California",
      "confidence": 0.847
    },
    ...
  ],
  "model_version": "step4-finetune-quick-cpu"
}
```

## Experiment Tracking

All runs logged to Weights & Biases:
- **Step 2:** EDA — clause distribution charts, context length histograms
- **Step 3:** Zero-shot baseline metrics
- **Step 4:** Training loss curve, span accuracy per epoch
- **Step 5:** Per-clause precision / recall / F1 table + charts

## Setup

```bash
git clone <repo>
cd contract-intelligence
pip install -r requirements.txt
cp .env.example .env  # add your WANDB_API_KEY

# Download CUAD dataset
python -m src.data.download_cuad

# Run EDA
python -m src.data.explore_cuad

# Fine-tune
python -m src.training.train --quick   # CPU
python -m src.training.train            # GPU (recommended)

# Evaluate
python -m src.training.evaluate

# Serve
docker compose up
```

## Trade-offs & Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Task framing | Span extraction (SQuAD) | Extracts actual clause text, not just yes/no |
| Long context | Sliding window (stride=128) | CUAD contracts average 6,000+ words; BERT limit is ~380 |
| Base model | Legal-BERT | Pre-trained on legal corpora — better than general BERT for contracts |
| Evaluation | Macro-F1 + has-answer split | Overall F1 is misleading due to no-answer majority |
| Deployment | Single-container Docker | Simple to run; swap model dir env var for a better checkpoint |

## Improving the model

The current weights are a CPU demo (50 training samples). To get production-quality results:

1. **Use Colab/GPU** — run `python -m src.training.train` (no `--quick` flag) with a T4 GPU; expect ~2 hours for full training
2. **Oversample positive windows** — filter `CUADDataset` to balance answer-containing vs empty windows
3. **Increase `max_windows_per_sample`** — set to `None` for full document coverage during training
