"""
Chat endpoint: run retrieve → reason → verify and return grounding metadata.

The reply text alone is not enough for legal prep. The UI should show
grounding_status and evidence so you can distrust fluent wrong answers.
"""

from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agents.prep_graph import prep_graph

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    matter_id: str = "bronner-2026"


class EvidenceOut(BaseModel):
    id: str
    source: str
    page: int | None = None
    source_type: str
    preview: str


class ChatResponse(BaseModel):
    reply: str
    matter_id: str
    grounding_status: str
    grounding_notes: str = ""
    evidence: list[EvidenceOut] = []
    claims_verified: int = 0
    claims_total: int = 0


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest):
    result = prep_graph.invoke(
        {
            "messages": [HumanMessage(content=body.message)],
            "matter_id": body.matter_id,
            "evidence": [],
        }
    )

    last = result["messages"][-1]
    text = getattr(last, "content", str(last))
    evidence_raw = result.get("evidence") or []
    claims = result.get("claims") or []
    verified = sum(1 for c in claims if c.get("verified"))

    evidence_out = [
        EvidenceOut(
            id=e["id"],
            source=e["source"],
            page=e.get("page"),
            source_type=e.get("source_type", "unknown"),
            preview=(e.get("text") or "")[:220],
        )
        for e in evidence_raw
    ]

    return ChatResponse(
        reply=text if isinstance(text, str) else str(text),
        matter_id=body.matter_id,
        grounding_status=result.get("grounding_status") or "unverified",
        grounding_notes=result.get("grounding_notes") or "",
        evidence=evidence_out,
        claims_verified=verified,
        claims_total=len(claims),
    )
