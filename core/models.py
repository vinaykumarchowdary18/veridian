"""
core/models.py — shared Pydantic models for the Veridian pipeline.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class EvidencePacket(BaseModel):
    query: str
    snippets: list[str]
    urls: list[str]
    raw_answer: Optional[str] = None


class Proposal(BaseModel):
    content: str
    model: str
    reasoning: Optional[str] = None


class CritiquePoint(BaseModel):
    severity: str        # "critical" | "major" | "minor"
    category: str        # "factual" | "logical" | "completeness" | "clarity" | "business_impact"
    description: str
    suggested_fix: Optional[str] = None


class CriticVerdict(BaseModel):
    critic_id: str
    model: str
    verdict: str         # "accept" | "revise" | "reject"
    score: float = Field(ge=0.0, le=1.0)
    critique_points: list[CritiquePoint]
    summary: str


class DebateRound(BaseModel):
    round_number: int
    proposal: Proposal
    critique_a: CriticVerdict
    critique_b: CriticVerdict
    arbiter_notes: Optional[str] = None


class ArbiterDecision(BaseModel):
    consensus_score: float = Field(ge=0.0, le=1.0)
    accepted_points: list[str]
    rejected_points: list[str]
    directive: str       # "finalize" | "revise" | "escalate"
    revised_proposal: Optional[str] = None
    reasoning: str


class IntelBrief(BaseModel):
    """The fully validated business intelligence output delivered to the user."""
    question: str
    executive_summary: str
    key_findings: list[str]
    risks_and_caveats: list[str]
    recommended_actions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    consensus_score: float
    debate_rounds: int
    sources: list[str]
    dissenting_points: list[str]
    models_used: dict[str, str]
    raw_answer: str      # full validated answer text
