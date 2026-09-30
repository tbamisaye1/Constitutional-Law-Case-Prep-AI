"""
Retrieve node: pull FAISS chunks for the latest user question.

If the index is empty, we still continue the graph, but abstain rules in
reason_node will refuse to invent an answer. That is intentional.
"""

from langchain_core.messages import HumanMessage

from app.agents.state import PrepState
from app.grounding.schemas import EvidenceHit, SourceType
from app.grounding.selection import extract_selection, retrieval_query
from app.rag.store import load_store


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


def retrieve_node(state: PrepState) -> dict:
    message = _latest_user_text(state)
    query = retrieval_query(message)
    selection = extract_selection(message)
    store = load_store()

    evidence: list[EvidenceHit] = []

    # Always put the highlight first so "what does this mean?" cannot be
    # answered from an unrelated Fourth Amendment chunk while ignoring Padilla.
    if selection:
        evidence.append(
            {
                "id": "ev-0",
                "text": selection,
                "source": "(selected passage)",
                "page": None,
                "source_type": "user_note",
                "score": 0.0,
            }
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

    # similarity_search_with_score returns (Document, score). Lower distance
    # is better for L2; we keep the raw score for abstain heuristics.
    pairs = store.similarity_search_with_score(query, k=5)
    for doc, score in pairs:
        meta = doc.metadata or {}
        source = str(meta.get("source", "unknown"))
        text = doc.page_content or ""
        # Skip near-duplicates of the selection we already injected.
        if selection and text.strip() == selection.strip():
            continue
        evidence.append(
            {
                "id": f"ev-{len(evidence)}",
                "text": text,
                "source": source,
                "page": meta.get("page"),
                "source_type": _guess_source_type(source),
                "score": float(score) if score is not None else None,
            }
        )

    notes = f"Retrieved {len(evidence)} passage(s)."
    if selection:
        notes += " Selected highlight is evidence[0] and drove the retrieval query."

    return {
        "evidence": evidence,
        "grounding_notes": notes,
    }
