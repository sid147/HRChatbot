# HR Resource Query Chatbot

## Overview
An intelligent HR assistant that answers natural-language resource allocation queries such as:
- "Find Python developers with 3+ years experience"
- "Who has worked on healthcare projects?"
- "Suggest people for a React Native project"
- "Find developers who know both AWS and Docker"

It implements a lightweight RAG (Retrieval-Augmentation-Generation) pipeline: semantic retrieval over employee profiles, augmentation with query context, and response generation (OpenAI optional; template fallback by default).

## Features
- Semantic employee search using sentence-transformers embeddings
- Natural language chat endpoint that recommends candidates with reasons
- Filtered employee search endpoint (skills, min experience, availability)
- Streamlit chat UI
- Gemini integration optional via `GEMINI_API_KEY`
- Clean FastAPI backend with auto docs

## Architecture
- Backend (FastAPI):
  - Data: `app/data/employees.json`
  - RAG components:
    - `app/rag/embeddings.py`: model loading and encoding
    - `app/rag/retriever.py`: query parsing, retrieval, scoring
    - `app/rag/generator.py`: Gemini-backed or template generator
  - Service: `app/services/search_service.py`
  - API: `app/main.py`
- Frontend (Streamlit): `frontend/streamlit_app.py`

Retrieval computes text embeddings of employee profiles and query using `all-MiniLM-L6-v2`, ranks by cosine similarity, and applies rule-based boosts for matched skills, domain keywords (e.g., healthcare), years of experience, and availability. Generation composes a concise recommendation; if `GEMINI_API_KEY` is set, it uses an LLM, otherwise a clear template.

## Setup & Installation

### Prerequisites
- Python 3.10+
- Internet access on first run to download the sentence-transformers model

### Backend
```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open docs at: http://127.0.0.1:8000/docs

Optional: set Google Gemini for richer generation (use a `.env` file at the repo root)
```bash
# .env contents (create a file named .env in the repo root)
# GEMINI_API_KEY=your-gemini-key
# GEMINI_MODEL=gemini-1.5-flash
```

### Frontend
```bash
cd frontend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install streamlit requests
$env:BACKEND_URL = "http://127.0.0.1:8000"
streamlit run streamlit_app.py
```

## API Documentation

### POST /chat
Request body:
```json
{
  "message": "I need someone experienced with machine learning for a healthcare project"
}
```
Response:
```json
{
  "query": "...",
  "reply": "...",
  "candidates": [
    {
      "employee": {"id": 3, "name": "...", "skills": ["..."], "experience_years": 6, "projects": ["..."], "availability": "available"},
      "score": 0.87,
      "reasons": ["skills: python, pytorch", "experience: 6 yrs", "domain: healthcare"]
    }
  ]
}
```

### GET /employees/search
Query params: `query`, `min_experience`, `skills` (comma-separated), `available_only`, `top_k`

Example:
```bash
curl -G "http://127.0.0.1:8000/employees/search" \
  --data-urlencode "query=Find Python developers with 3+ years" \
  --data-urlencode "min_experience=3" \
  --data-urlencode "skills=Python, Docker"
```

## AI Development Process
- Assistants used: Cursor (this environment), optional ChatGPT for ideation
- How AI helped:
  - Sketched project structure and key modules
  - Generated boilerplate FastAPI endpoints and Streamlit UI
  - Implemented retrieval heuristics and prompt scaffolding
- Estimated AI-assisted code: ~70%; hand-written/refined: ~30%
- Interesting outcomes: Hybrid scoring (semantic + rule boosts) provided good relevance without a full vector DB
- Challenges: Keeping dependencies light and ensuring Windows-friendly setup

## Technical Decisions
- Sentence-transformers local embeddings for privacy and cost; no vector DB needed for small dataset
- Gemini optional: richer natural responses when key is provided; falls back to deterministic summary otherwise
- FastAPI chosen for speed, typing, and docs; Streamlit for quickest chat UI

## Future Improvements
- Persist embeddings and add FAISS for larger datasets
- Advanced query parser (skills synonyms, fuzzy matching)
- Add profile details view and availability calendar integration
- Authentication and role-based access
- Dockerized deployment

## Demo
- Local demo: run backend and frontend as above
- Optionally deploy Streamlit to Streamlit Cloud and backend to Render/Fly.io 