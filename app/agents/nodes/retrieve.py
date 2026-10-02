"""
Retrieve node: pull FAISS chunks for the latest user question.

If the index is empty, we still continue the graph, but abstain rules in
reason_node will refuse to invent an answer. That is intentional.
"""

from langchain_core.messages import HumanMessage

from app.agents.state import PrepState
from app.grounding.schemas import EvidenceHit, SourceType
from app.grounding.selection import (
    extract_selection,
    extract_source_file,
    extract_source_page,
    retrieval_query,
)
from app.rag.store import docs_for_source, load_store


def _guess_source_type(source: str) -> SourceType:
    name = source.lower()
    if "record" in name or name.startswith("r.") or "bronner" in name and "guide" not in name:
        return "record"
    if "note" in name or source.startswith("(selected"):
        return "user_note"
    if "law review" in name or "nyu" in name or "villanova" in name:
        return "secondary"
    return "precedent"


def _latest_user_text(state: PrepState) -> str:
    for msg in reversed(list(state["messages"])):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _append_hit(
    evidence: list[EvidenceHit],
    *,
    text: str,
    source: str,
    page,
    source_type: SourceType,
    score: float | None,
) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        return
    # Skip near-duplicates already in the list.
    for existing in evidence:
        if (existing.get("text") or "").strip() == cleaned:
            return
    evidence.append(
        {
            "id": f"ev-{len(evidence)}",
            "text": cleaned,
            "source": source,
            "page": page,
            "source_type": source_type,
            "score": float(score) if score is not None else None,
        }
    )


def retrieve_node(state: PrepState) -> dict:
    message = _latest_user_text(state)
    query = retrieval_query(message)
    selection = extract_selection(message)
    source_file = extract_source_file(message)
    source_page = extract_source_page(message)
    store = load_store()

    evidence: list[EvidenceHit] = []

    # Always put the highlight first so "what does this mean?" cannot be
    # answered from an unrelated Fourth Amendment chunk while ignoring Padilla.
    if selection:
        label = source_file or "(selected passage)"
        _append_hit(
            evidence,
            text=selection,
            source=label,
            page=source_page,
            source_type="user_note",
            score=0.0,
        )

    if store is None or not query.strip():
        if evidence:
            return {
                "evidence": evidence,
                "grounding_notes": "Using the selected passage only (index empty or query blank).",
            }
        return {
            "evidence": [],
            "grounding_status": "no_evidence",
            "grounding_notes": "No articles indexed yet, or empty question. Upload PDFs via /ingest/pdf.",
        }

    # When we know which article the highlight came from, pull nearby pages from
    # that PDF first so reading-mode questions stay in-article.
    same_source_hits = 0
    if source_file:
        for doc, score in docs_for_source(
            store, source_file, page=source_page, page_window=1, limit=5
        ):
            meta = doc.metadata or {}
            _append_hit(
                evidence,
                text=doc.page_content or "",
                source=str(meta.get("source", source_file)),
                page=meta.get("page"),
                source_type=_guess_source_type(str(meta.get("source", source_file))),
                score=score,
            )
            same_source_hits += 1

    # similarity_search_with_score returns (Document, score). Lower distance
    # is better for L2; we keep the raw score for abstain heuristics.
    pairs = store.similarity_search_with_score(query, k=6)
    for doc, score in pairs:
        meta = doc.metadata or {}
        source = str(meta.get("source", "unknown"))
        _append_hit(
            evidence,
            text=doc.page_content or "",
            source=source,
            page=meta.get("page"),
            source_type=_guess_source_type(source),
            score=float(score) if score is not None else None,
        )

    # Cap so the reason prompt stays readable.
    evidence = evidence[:8]

    notes = f"Retrieved {len(evidence)} passage(s)."
    if selection:
        notes += " Selected highlight is evidence[0] and drove the retrieval query."
    if source_file:
        notes += f" Preferred source article: {source_file}"
        if source_page is not None:
            notes += f" p.{source_page}"
        notes += f" ({same_source_hits} nearby chunk(s))."

    return {
        "evidence": evidence,
        "grounding_notes": notes,
    }
