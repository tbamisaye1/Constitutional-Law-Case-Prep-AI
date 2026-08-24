"""
Retrieve node: pull FAISS chunks for the latest user question.

If the index is empty, we still continue the graph, but abstain rules in
reason_node will refuse to invent an answer. That is intentional.
"""

from langchain_core.messages import HumanMessage

from app.agents.state import PrepState
from app.grounding.schemas import EvidenceHit, SourceType
from app.rag.store import load_store


def _guess_source_type(source: str) -> SourceType:
    name = source.lower()
    if "record" in name or name.startswith("r.") or "bronner" in name and "guide" not in name:
        return "record"
    if "note" in name:
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
    question = _latest_user_text(state)
    store = load_store()
    if store is None or not question.strip():
        return {
            "evidence": [],
            "grounding_status": "no_evidence",
            "grounding_notes": "No articles indexed yet, or empty question. Upload PDFs via /ingest/pdf.",
        }

    # similarity_search_with_score returns (Document, score). Lower distance
    # is better for L2; we keep the raw score for abstain heuristics.
    pairs = store.similarity_search_with_score(question, k=5)
    evidence: list[EvidenceHit] = []
    for i, (doc, score) in enumerate(pairs):
        meta = doc.metadata or {}
        source = str(meta.get("source", "unknown"))
        evidence.append(
            {
                "id": f"ev-{i}",
                "text": doc.page_content,
                "source": source,
                "page": meta.get("page"),
                "source_type": _guess_source_type(source),
                "score": float(score) if score is not None else None,
            }
        )

    return {
        "evidence": evidence,
        "grounding_notes": f"Retrieved {len(evidence)} passages from uploaded articles.",
    }
