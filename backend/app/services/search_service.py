from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models import CandidateMatch, Employee
from app.rag.embeddings import (
    initialize_fallback_tfidf,
)
from app.rag.retriever import (
    build_employee_embeddings,
    score_candidates,
)

_EMPLOYEES: List[Employee] = []
_EMP_EMBEDDINGS: Optional[np.ndarray] = None
_KNOWN_SKILLS: List[str] = []


def _load_employees(data_path: str) -> List[Employee]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    employees = [Employee(**e) for e in raw["employees"]]
    return employees


def initialize(data_path: str) -> None:
    global _EMPLOYEES, _EMP_EMBEDDINGS, _KNOWN_SKILLS
    _EMPLOYEES = _load_employees(data_path)

    # Build skill universe
    skill_set = set()
    for e in _EMPLOYEES:
        for s in e.skills:
            skill_set.add(s.lower())
    _KNOWN_SKILLS = sorted(skill_set)

    # Build embeddings
    corpus, embeddings = build_employee_embeddings(_EMPLOYEES)
    _EMP_EMBEDDINGS = embeddings

    # Initialize TF-IDF fallback with corpus text so searches still work without transformers
    try:
        initialize_fallback_tfidf(corpus)
    except Exception:
        pass


def get_state_snapshot() -> Dict:
    return {
        "num_employees": len(_EMPLOYEES),
        "num_skills": len(_KNOWN_SKILLS),
        "have_embeddings": _EMP_EMBEDDINGS is not None,
        "gemini": {
            "enabled": bool(os.getenv("GEMINI_API_KEY")),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        },
    }


def search_by_query(query: str, top_k: int = 10) -> Tuple[List[CandidateMatch], Dict]:
    if _EMP_EMBEDDINGS is None:
        raise RuntimeError("Service not initialized")
    candidates, meta = score_candidates(
        query=query,
        employees=_EMPLOYEES,
        employee_embeddings=_EMP_EMBEDDINGS,
        known_skills=_KNOWN_SKILLS,
        top_k=top_k,
    )
    return candidates, meta


def search_with_filters(
    query: Optional[str] = None,
    min_experience: Optional[int] = None,
    skills: Optional[List[str]] = None,
    available_only: bool = False,
    top_k: int = 20,
) -> Tuple[List[CandidateMatch], Dict]:
    q = query or ""
    candidates, meta = search_by_query(q, top_k=top_k)

    filtered: List[CandidateMatch] = []
    for c in candidates:
        emp = c.employee
        if min_experience is not None and emp.experience_years < min_experience:
            continue
        if skills:
            emp_skills_lower = {s.lower() for s in emp.skills}
            required = {s.lower() for s in skills}
            if not required.issubset(emp_skills_lower):
                continue
        if available_only and not str(emp.availability).lower().startswith("avail"):
            continue
        filtered.append(c)

    meta.update({
        "filters_applied": {
            "min_experience": min_experience,
            "skills": skills,
            "available_only": available_only,
        }
    })
    return filtered, meta


def list_employees() -> List[Employee]:
    return list(_EMPLOYEES) 