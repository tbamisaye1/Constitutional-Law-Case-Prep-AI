"""
Chat endpoint: run retrieve → reason → verify and return grounding metadata.

The reply text alone is not enough for legal prep. The UI should show
grounding_status and evidence so you can distrust fluent wrong answers.

grounding_source:
  documents  — FAISS RAG only (default; unchanged)
  web_plus   — corpus + OpenRouter web search
"""

from typing import Literal

from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agents.prep_graph import prep_graph
from app.grounding.selection import build_chat_message
from app.llm.openrouter import normalize_model_tier

router = APIRouter(prefix="/chat", tags=["chat"])

GroundingSourceIn = Literal["documents", "web_plus"]
ModelTierIn = Literal["standard", "advanced"]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class NotebookNoteIn(BaseModel):
    """A short local note chunk searched in the browser (not uploaded to FAISS).

    Covers OneNote notebook pages, PDF annotations, and case-library tabs.
    """

    id: str = "note"
    title: str = "Untitled note"
    text: str = Field(min_length=1)
    section_name: str | None = None
    page_id: str | None = None
    notes_path: str | None = None
    source_type: str | None = None
    page: int | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    matter_id: str = "bronner-2026"
    grounding_source: GroundingSourceIn = "documents"
    # standard = gpt-4o-mini; advanced = gpt-5-mini (Ask AI Advanced toggle).
    model_tier: ModelTierIn = "standard"
    # Highlighted PDF / guide text from the Ask AI bubble. Optional for older
    # clients; when present it is packed into the graph message so retrieval
    # and the model actually see what the user selected.
    selection: str | None = None
    # Which uploaded PDF the highlight came from (filename as stored in FAISS).
    source_file: str | None = None
    # 1-based PDF page of the highlight, when known.
    page: int | None = None
    # Prior turns in this Ask AI thread (not including the current message).
    history: list[ChatTurn] | None = None
    # Opt-in notebook evidence from this browser's localStorage notebook.
    notes: list[NotebookNoteIn] | None = None


class EvidenceOut(BaseModel):
    id: str
    source: str
    page: int | None = None
    source_type: str
    preview: str
    url: str | None = None
    notes_path: str | None = None


class ChatResponse(BaseModel):
    reply: str
    matter_id: str
    grounding_status: str
    grounding_notes: str = ""
    grounding_source: GroundingSourceIn = "documents"
    model_tier: ModelTierIn = "standard"
    evidence: list[EvidenceOut] = []
    claims_verified: int = 0
    claims_total: int = 0


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest):
    source = body.grounding_source or "documents"
    tier = normalize_model_tier(body.model_tier)
    packed = build_chat_message(
        body.message,
        body.selection,
        source_file=body.source_file,
        page=body.page,
    )
    history = [
        {"role": turn.role, "content": turn.content.strip()}
        for turn in (body.history or [])
        if turn.content and turn.content.strip()
    ][-12:]
    client_notes = [
        {
            "id": n.id,
            "title": n.title,
            "text": n.text.strip(),
            "section_name": n.section_name,
            "page_id": n.page_id,
            "notes_path": n.notes_path,
            "source_type": n.source_type,
            "page": n.page,
        }
        for n in (body.notes or [])
        if n.text and n.text.strip()
    ][:8]
    # Log enough to debug "answered the wrong case" without dumping full PDFs.
    sel = (body.selection or "").strip()
    src = (body.source_file or "").strip()
    print(
        f"[chat] source={source} tier={tier} matter={body.matter_id} "
        f"q_len={len(body.message)} sel_len={len(sel)} "
        f"file={src!r} page={body.page!r} hist={len(history)} "
        f"notes={len(client_notes)} "
        f"q_preview={body.message[:120]!r} "
        f"sel_preview={sel[:160]!r}"
    )
    result = prep_graph.invoke(
        {
            "messages": [HumanMessage(content=packed)],
            "matter_id": body.matter_id,
            "grounding_source": source,
            "model_tier": tier,
            "chat_history": history,
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
        EvidenceOut(
            id=e["id"],
            source=e["source"],
            page=e.get("page"),
            source_type=e.get("source_type", "unknown"),
            preview=(e.get("text") or "")[:220],
            url=e.get("url"),
            notes_path=e.get("notes_path"),
        )
        for e in evidence_raw
    ]

    return ChatResponse(
        reply=text if isinstance(text, str) else str(text),
        matter_id=body.matter_id,
        grounding_status=result.get("grounding_status") or "unverified",
        grounding_notes=result.get("grounding_notes") or "",
        grounding_source=source,
        model_tier=tier,
        evidence=evidence_out,
        claims_verified=verified,
        claims_total=len(claims),
    )
