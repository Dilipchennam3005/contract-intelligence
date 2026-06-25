"""
Contract Intelligence API
POST /analyze  — run all 41 clause checks on a contract
GET  /clauses  — list all clause types
GET  /health   — liveness check

Run locally:  uvicorn api.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.model import CLAUSE_TYPES, DEVICE, MODEL_DIR, contract_model
from api.schemas import AnalyzeResponse, ClauseResult, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    contract_model.load()
    yield


app = FastAPI(
    title="Contract Intelligence API",
    description="Fine-tuned Legal-BERT for 41 CUAD clause types",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text:  str
    title: str = "Untitled Contract"

    model_config = {"json_schema_extra": {"example": {
        "title": "Software License Agreement",
        "text":  "THIS SOFTWARE LICENSE AGREEMENT is entered into as of January 1, 2024..."
    }}}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=contract_model.loaded,
        model_path=str(MODEL_DIR),
        device=DEVICE,
    )


@app.get("/clauses", tags=["Model"])
def list_clauses():
    """Return all 41 CUAD clause types this model can identify."""
    return {"clause_types": CLAUSE_TYPES, "total": len(CLAUSE_TYPES)}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
def analyze(request: AnalyzeRequest):
    """
    Run all 41 clause checks on the submitted contract text.
    Returns extracted spans and confidence scores for each clause type.
    """
    if not contract_model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if len(request.text.strip()) < 50:
        raise HTTPException(status_code=422, detail="Contract text too short (min 50 chars)")

    raw = contract_model.analyze(request.text, request.title)

    return AnalyzeResponse(
        contract_title=raw["contract_title"],
        total_clauses_checked=raw["total_clauses_checked"],
        clauses_found=raw["clauses_found"],
        results=[ClauseResult(**r) for r in raw["results"]],
        model_version=raw["model_version"],
    )
