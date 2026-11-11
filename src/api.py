"""Day 10: REST API.

Run:  uvicorn src.api:app --reload      then open http://localhost:8000/docs
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chain import answer
from src.retriever import list_policies

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-api")

app = FastAPI(title="Insurance Policy RAG Assistant", version="1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    policy: str | None = None


class Source(BaseModel):
    policy: str | None
    page: int | None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/policies")
def policies():
    return {"policies": list_policies()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.perf_counter()
    try:
        result = answer(req.question, policy=req.policy)
    except Exception as exc:  # keep the API up and log the cause
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency = int((time.perf_counter() - start) * 1000)
    log.info("q=%r policy=%s latency_ms=%d", req.question, req.policy, latency)
    return AskResponse(**result, latency_ms=latency)
