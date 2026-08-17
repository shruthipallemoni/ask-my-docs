"""
main.py

FastAPI service exposing the RAG pipeline as a real HTTP API, plus a
minimal chat UI served from the same origin (no CORS setup needed, since
the frontend and API share one server).

Run (from project root, Qdrant running + GROQ_API_KEY set):
    uvicorn src.api.main:app --reload

Then open http://localhost:8000/ for the chat UI, or
http://localhost:8000/docs for interactive Swagger docs, or POST directly:
    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"question": "how do I create an issue"}'
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.generation.rag_graph import run_rag_pipeline

app = FastAPI(
    title="Ask My Docs — GitHub API Assistant",
    description="RAG over GitHub's official REST API documentation, with enforced source citations.",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="A developer question about GitHub's REST API")
    retrieve_k: int = Field(25, ge=5, le=100, description="How many candidates to retrieve before reranking")
    rerank_top_k: int = Field(3, ge=1, le=10, description="How many top reranked chunks to expand into context")


class QueryResponse(BaseModel):
    question: str
    answer: str
    citation_valid: bool
    regeneration_attempts: int
    sources: list[str]  # endpoint_ids the answer was grounded in
    latency_seconds: float


@app.get("/")
def chat_ui():
    """Serves the minimal chat UI. Not under /docs — that's FastAPI's own Swagger UI, left untouched."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Basic liveness check — does NOT verify Qdrant/Groq are reachable, just that the API is up."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.perf_counter()

    try:
        result = run_rag_pipeline(
            request.question,
            retrieve_k=request.retrieve_k,
            rerank_top_k=request.rerank_top_k,
        )
    except RuntimeError as e:
        # e.g. GROQ_API_KEY not set — surface as a clear 500, not a silent crash
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.perf_counter() - start

    return QueryResponse(
        question=request.question,
        answer=result["display_answer"],
        citation_valid=result["citation_valid"],
        regeneration_attempts=result["regeneration_attempts"],
        sources=sorted(result["expanded_endpoint_ids"]),
        latency_seconds=round(latency, 2),
    )