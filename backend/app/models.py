from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Availability(str, Enum):
    available = "available"
    busy = "busy"
    soon = "soon"


class Employee(BaseModel):
    id: int
    name: str
    skills: List[str]
    experience_years: int = Field(ge=0)
    projects: List[str]
    availability: Availability

    @field_validator("skills", mode="before")
    @classmethod
    def normalize_skills(cls, value: List[str]) -> List[str]:
        return [skill.strip() for skill in value]


class CandidateMatch(BaseModel):
    employee: Employee
    score: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    query: str
    reply: str
    candidates: List[CandidateMatch]


class SearchResponse(BaseModel):
    candidates: List[CandidateMatch]
    total: int
    query: Optional[str] = None
    filters: Optional[dict] = None 