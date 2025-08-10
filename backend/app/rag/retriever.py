from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.models import CandidateMatch, Employee
from app.rag.embeddings import compute_embeddings, embed_text


DOMAIN_KEYWORDS = {
    "healthcare",
    "finance",
    "fintech",
    "e-commerce",
    "ecommerce",
    "education",
    "logistics",
    "retail",
    "saas",
    "iot",
    "gaming",
    "banking",
    "telecom",
    "energy",
    "travel",
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def build_profile_text(employee: Employee) -> str:
    parts = [
        employee.name,
        f"{employee.experience_years} years experience",
        ", ".join(employee.skills),
        ", ".join(employee.projects),
        employee.availability,
    ]
    return normalize_text(" | ".join(parts))


def parse_query(query: str, known_skills: Iterable[str]) -> Dict:
    q = normalize_text(query)

    # years of experience
    min_years = 0
    years_match = re.search(r"(\d+)\+?\s*(years?|yrs?)", q)
    if years_match:
        try:
            min_years = int(years_match.group(1))
        except Exception:
            min_years = 0

    # skills
    known_skills_lower = {s.lower() for s in known_skills}
    tokens = re.split(r"[^a-z0-9\+#\.]+", q)
    found_skills = set()
    for token in tokens:
        if not token:
            continue
        # handle variations like react-native, node.js
        normalized = token.lower()
        if normalized in known_skills_lower:
            found_skills.add(normalized)
        elif normalized == "react" and "react" in known_skills_lower:
            found_skills.add("react")
        elif normalized in {"aws", "gcp", "azure", "docker", "kubernetes", "k8s"} and normalized in known_skills_lower:
            found_skills.add(normalized)

    # domain
    found_domains = [d for d in DOMAIN_KEYWORDS if d in q]

    available_only = any(kw in q for kw in ["available", "availability", "free now", "capacity"])

    return {
        "min_years": min_years,
        "skills": sorted(found_skills),
        "domains": found_domains,
        "available_only": available_only,
    }


def compute_similarity_matrix(query_embedding: np.ndarray, employee_embeddings: np.ndarray) -> np.ndarray:
    return cosine_similarity(query_embedding.reshape(1, -1), employee_embeddings)[0]


def score_candidates(
    query: str,
    employees: List[Employee],
    employee_embeddings: np.ndarray,
    known_skills: Iterable[str],
    top_k: int = 10,
) -> Tuple[List[CandidateMatch], Dict]:
    if len(employees) == 0:
        return [], {"parsed": {}}

    parsed = parse_query(query, known_skills)

    # Semantic similarity
    query_embedding = embed_text(normalize_text(query))
    sim = compute_similarity_matrix(query_embedding, employee_embeddings)

    # Rule-based boosts
    boosts = np.zeros_like(sim)

    for idx, emp in enumerate(employees):
        reasons: List[str] = []
        boost = 0.0

        # Skills match boosts
        emp_skills_lower = {s.lower() for s in emp.skills}
        matched_skills = [s for s in parsed["skills"] if s in emp_skills_lower]
        if matched_skills:
            per_skill = 0.05
            boost += min(0.20, per_skill * len(matched_skills))
            reasons.append(f"skills match: {', '.join(matched_skills)}")

        # Experience boost
        if parsed["min_years"] and emp.experience_years >= parsed["min_years"]:
            boost += 0.05
            reasons.append(f"experience >= {parsed['min_years']} years")

        # Domain boost
        if parsed["domains"]:
            emp_projects_text = ", ".join(emp.projects).lower()
            matched_domains = [d for d in parsed["domains"] if d in emp_projects_text]
            if matched_domains:
                boost += 0.10
                reasons.append(f"domain: {', '.join(matched_domains)}")

        # Availability boost
        if parsed["available_only"] and emp.availability.lower().startswith("avail"):
            boost += 0.05
            reasons.append("available now")

        boosts[idx] = boost

    combined = (0.8 * sim) + boosts

    # Build matches with reasons
    matches: List[Tuple[int, float, List[str]]] = []
    for idx, score in enumerate(combined):
        matches.append((idx, float(max(0.0, min(1.0, score))), []))

    # Sort and format
    top_indices = np.argsort(-combined)[: top_k]

    candidate_matches: List[CandidateMatch] = []
    for i in top_indices:
        emp = employees[i]
        reasons: List[str] = []
        emp_skills_lower = {s.lower() for s in emp.skills}
        matched_skills = [s for s in parsed["skills"] if s in emp_skills_lower]
        if matched_skills:
            reasons.append(f"skills: {', '.join(matched_skills)}")
        if parsed["min_years"] and emp.experience_years >= parsed["min_years"]:
            reasons.append(f"experience: {emp.experience_years} yrs")
        if parsed["domains"]:
            emp_projects_text = ", ".join(emp.projects).lower()
            matched_domains = [d for d in parsed["domains"] if d in emp_projects_text]
            if matched_domains:
                reasons.append(f"domain: {', '.join(matched_domains)}")
        if parsed["available_only"] and emp.availability.lower().startswith("avail"):
            reasons.append("available")

        candidate_matches.append(
            CandidateMatch(employee=emp, score=float(combined[i]), reasons=reasons)
        )

    return candidate_matches, {"parsed": parsed}


def build_employee_embeddings(employees: List[Employee]) -> Tuple[List[str], np.ndarray]:
    corpus = [build_profile_text(emp) for emp in employees]
    embeddings = compute_embeddings(corpus)
    return corpus, embeddings 