"""
Shared prep-agent invoke used by POST /chat and the MCP ask_prep_agent tool.

Keep request packing and graph invocation in one place so the HTTP and MCP
paths cannot drift.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage

from app.agents.prep_graph import prep_graph
from app.grounding.selection import build_chat_message
from app.llm.openrouter import normalize_model_tier

GroundingSource = Literal["documents", "web_plus"]
ModelTier = Literal["standard", "advanced"]


def run_prep_chat(
    message: str,
    *,
    matter_id: str = "",
    grounding_source: GroundingSource = "documents",
    model_tier: ModelTier = "standard",
    selection: str | None = None,
    source_file: str | None = None,
    page: int | None = None,
    history: list[dict[str, str]] | None = None,
    notes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Run retrieve → reason → verify and return a plain dict for either transport.

    Returns:
        reply, matter_id, grounding_status, grounding_notes, grounding_source,
        model_tier, evidence, claims_verified, claims_total.
    """
    source: GroundingSource = grounding_source or "documents"
    tier = normalize_model_tier(model_tier)
    packed = build_chat_message(
        message,
        selection,
        source_file=source_file,
        page=page,
    )
    history_rows = [
        {"role": turn["role"], "content": turn["content"].strip()}
        for turn in (history or [])
        if turn.get("content") and str(turn["content"]).strip()
        and turn.get("role") in ("user", "assistant")
    ][-12:]
    client_notes = [
        {
            "id": n.get("id") or "note",
            "title": n.get("title") or "Untitled note",
            "text": str(n.get("text") or "").strip(),
            "section_name": n.get("section_name"),
            "page_id": n.get("page_id"),
            "notes_path": n.get("notes_path"),
            "source_type": n.get("source_type"),
            "page": n.get("page"),
        }
        for n in (notes or [])
        if n.get("text") and str(n.get("text")).strip()
    ][:8]

    result = prep_graph.invoke(
        {
            "messages": [HumanMessage(content=packed)],
            "matter_id": matter_id or "",
            "grounding_source": source,
            "model_tier": tier,
            "chat_history": history_rows,
            "client_notes": client_notes,
            "evidence": [],
        }
    )

    last = result["messages"][-1]
    text = getattr(last, "content", str(last))
    evidence_raw = result.get("evidence") or []
    claims = result.get("claims") or []
    verified = sum(1 for c in claims if c.get("verified"))

    evidence_out = [
        {
            "id": e["id"],
            "source": e["source"],
            "page": e.get("page"),
            "source_type": e.get("source_type", "unknown"),
            "preview": (e.get("text") or "")[:220],
            "url": e.get("url"),
            "notes_path": e.get("notes_path"),
        }
        for e in evidence_raw
    ]

    return {
        "reply": text if isinstance(text, str) else str(text),
        "matter_id": matter_id or "",
        "grounding_status": result.get("grounding_status") or "unverified",
        "grounding_notes": result.get("grounding_notes") or "",
        "grounding_source": source,
        "model_tier": tier,
        "evidence": evidence_out,
        "claims_verified": verified,
        "claims_total": len(claims),
    }
