"""
Shapes for grounded legal answers.

Why dataclasses/dicts instead of free prose: each claim must point at
evidence. That makes the verify node mechanical (string match) instead of
"trust the model that it cited something."
"""

from typing import Literal, NotRequired, TypedDict


SourceType = Literal["record", "precedent", "secondary", "user_note", "web", "unknown"]

GroundingStatus = Literal["grounded", "partial", "abstained", "unverified", "no_evidence"]


class EvidenceHit(TypedDict):
    """One retrieved chunk the model is allowed to use."""

    id: str
    text: str
    source: str
    page: int | None
    source_type: SourceType
    score: float | None
    # Present when source_type is "web" (OpenRouter url citation).
    url: NotRequired[str]


class Claim(TypedDict):
    """A single assertable sentence plus the evidence it rests on."""

    claim: str
    support_ids: list[str]
    quote: str
    # Set by verify node: True if quote (approx) appears in supported chunks.
    verified: bool | None


class GroundingReport(TypedDict):
    status: GroundingStatus
    claims: list[Claim]
    evidence_ids: list[str]
    notes: str
