from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.models import CandidateMatch, ChatRequest, ChatResponse, SearchResponse
from app.rag.generator import generate_response
from app.services import search_service

# Load environment variables from nearest .env if present
load_dotenv(find_dotenv(), override=False)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "employees.json"

app = FastAPI(title="HR Resource Query Chatbot", version="0.1.0")

# Allow local dev origins (FastAPI docs, Streamlit localhost variants)
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    if not DATA_PATH.exists():
        raise RuntimeError(f"Employees data not found at {DATA_PATH}")
    search_service.initialize(str(DATA_PATH))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "state": search_service.get_state_snapshot()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.message:
        raise HTTPException(status_code=400, detail="message is required")
    candidates, meta = search_service.search_by_query(req.message, top_k=10)
    reply = generate_response(req.message, candidates)
    return ChatResponse(query=req.message, reply=reply, candidates=candidates)


@app.get("/employees/search", response_model=SearchResponse)
async def employee_search(
    query: Optional[str] = Query(None, description="Natural language query"),
    min_experience: Optional[int] = Query(None, ge=0),
    skills: Optional[str] = Query(None, description="Comma-separated skills list"),
    available_only: bool = Query(False),
    top_k: int = Query(20, ge=1, le=50),
) -> SearchResponse:
    skills_list = [s.strip() for s in skills.split(",")] if skills else None
    candidates, meta = search_service.search_with_filters(
        query=query,
        min_experience=min_experience,
        skills=skills_list,
        available_only=available_only,
        top_k=top_k,
    )
    return SearchResponse(candidates=candidates, total=len(candidates), query=query, filters=meta.get("filters_applied")) 