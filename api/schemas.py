from pydantic import BaseModel


class ClauseResult(BaseModel):
    clause_type: str
    found: bool
    extracted_text: str | None
    confidence: float


class AnalyzeResponse(BaseModel):
    contract_title: str
    total_clauses_checked: int
    clauses_found: int
    results: list[ClauseResult]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
