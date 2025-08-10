from __future__ import annotations

import os
import logging
from typing import List

from app.models import CandidateMatch

logger = logging.getLogger("rag.generator")


essential_max_len = 10000


def _format_candidate_bullets(candidates: List[CandidateMatch]) -> str:
    lines = []
    for c in candidates:
        emp = c.employee
        reason_text = f" | reasons: {', '.join(c.reasons)}" if c.reasons else ""
        lines.append(
            f"- {emp.name} — {emp.experience_years} yrs; skills: {', '.join(emp.skills)}; projects: {', '.join(emp.projects)}; availability: {emp.availability}{reason_text}"
        )
    text = "\n".join(lines)
    return text if len(text) <= essential_max_len else text[:essential_max_len]


def _extract_text(result) -> str:
    # google-generativeai returns various shapes; prefer result.text
    try:
        if hasattr(result, "text") and result.text:
            return str(result.text)
        # Try candidates -> content -> parts
        candidates = getattr(result, "candidates", None)
        if candidates:
            parts = []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    txt = getattr(part, "text", None)
                    if txt:
                        parts.append(str(txt))
            if parts:
                return "\n".join(parts)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to extract Gemini response text: %s", e)
    return ""


def generate_response(query: str, candidates: List[CandidateMatch]) -> str:
    # Try Google Gemini if key is present
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and len(candidates) > 0:
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            model = genai.GenerativeModel(model_name)

            top = candidates[:5]
            context = _format_candidate_bullets(top)
            prompt = (
                "You are an HR assistant. Recommend employees based on the user query and provided candidates.\n"
                f"User query: {query}\n\nCandidates:\n{context}\n\n"
                "Write a concise recommendation (6-10 sentences). Highlight the top 2-3 choices with specific reasons (skills, domain, experience, availability)."
            )
            result = model.generate_content(prompt)
            text = _extract_text(result)
            if text:
                return text
            logger.warning("Gemini returned no text. Model=%s", model_name)
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            # Fallback to template below on any error
            pass

    if len(candidates) == 0:
        return (
            "I could not find matching employees for your request. Try adjusting skills, years, or domain keywords."
        )

    top = candidates[:5]
    bullets = _format_candidate_bullets(top)
    return (
        f"Based on your query, here are suitable candidates:\n\n{bullets}\n\n"
        "Would you like me to check their availability or set up introductions?"
    ) 