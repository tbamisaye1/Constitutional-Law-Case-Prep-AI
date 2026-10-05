"""
Retrieve node: pull FAISS chunks for the latest user question.

Also merges optional client notebook notes (browser-local) that Ask AI
sent with the request when the user opted into "Include my notes".

If the index is empty and there are no notes, we still continue the graph,
but abstain rules in reason_node will refuse to invent an answer.
"""

from langchain_core.messages import HumanMessage

from app.agents.state import PrepState
from app.grounding.schemas import EvidenceHit, SourceType
from app.grounding.selection import (
    extract_selection,
    extract_source_file,
    extract_source_page,
    mentions_instant_case,
    retrieval_query,
)
from app.rag.store import docs_for_instant_case, docs_for_source, is_instant_case_source, load_store


def _guess_source_type(source: str) -> SourceType:
    name = source.lower()
    if is_instant_case_source(source):
        return "record"
    if "record" in name or name.startswith("r."):
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
    url: str | None = None,
    notes_path: str | None = None,
) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        return
    # Skip near-duplicates already in the list.
    for existing in evidence:
        if (existing.get("text") or "").strip() == cleaned:
            return
    hit: EvidenceHit = {
        "id": f"ev-{len(evidence)}",
        "text": cleaned,
        "source": source,
        "page": page,
        "source_type": source_type,
        "score": float(score) if score is not None else None,
    }
    if url:
        hit["url"] = url
    if notes_path:
        hit["notes_path"] = notes_path
    evidence.append(hit)


_LOCAL_NOTE_TYPES = frozenset({"notebook", "annotation", "user_note"})


def _merge_client_notes(evidence: list[EvidenceHit], client_notes: list[dict] | None) -> int:
    """Append browser-searched notebook pages / PDF annotations. Returns how many were added."""
    added = 0
    for note in client_notes or []:
        text = str(note.get("text") or "").strip()
        if not text:
            continue
        title = str(note.get("title") or "Untitled note").strip() or "Untitled note"
        section = str(note.get("section_name") or "").strip()
        raw_type = str(note.get("source_type") or "notebook").strip().lower()
        source_type = raw_type if raw_type in _LOCAL_NOTE_TYPES else "notebook"
        page_raw = note.get("page")
        page = int(page_raw) if isinstance(page_raw, int) and page_raw > 0 else None
        if source_type == "annotation":
            label = f"{title} (Annotation · {section})" if section else f"{title} (Annotation)"
        else:
            label = f"{title} (Notes · {section})" if section else f"{title} (Notes)"
        path = str(note.get("notes_path") or "").strip() or None
        before = len(evidence)
        _append_hit(
            evidence,
            text=text,
            source=label,
            page=page,
            source_type=source_type,
            score=0.0,
            notes_path=path,
        )
        if len(evidence) > before:
            added += 1
    return added


def retrieve_node(state: PrepState) -> dict:
    message = _latest_user_text(state)
    query = retrieval_query(message)
    selection = extract_selection(message)
    source_file = extract_source_file(message)
    source_page = extract_source_page(message)
    store = load_store()
    client_notes = list(state.get("client_notes") or [])

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

    # Notebook pages + PDF annotations the browser matched locally (opt-in).
    # Prefer early so "what did I write about NDAA" is not drowned by FAISS hits.
    notes_added = _merge_client_notes(evidence, client_notes)

    if store is None or not query.strip():
        if evidence:
            note_bit = (
                f" Including {notes_added} local note/annotation chunk(s)."
                if notes_added
                else ""
            )
            return {
                "evidence": evidence,
                "grounding_notes": (
                    "Using selected passage and/or your notes/annotations only "
                    f"(index empty or query blank).{note_bit}"
                ),
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
    # Uploaded docs mode searches the whole index (Instant Case + Case library
    # + Articles), not an Articles-only shelf.
    pairs = store.similarity_search_with_score(query, k=8)
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

    # Instant Case questions: force query-ranked record chunks in (not caption
    # pages 1–2). Early-page bias previously made every Bronner ask look empty.
    instant_hits = 0
    if mentions_instant_case(query) or mentions_instant_case(message):
        for doc, score in docs_for_instant_case(store, query, limit=10):
            meta = doc.metadata or {}
            before = len(evidence)
            _append_hit(
                evidence,
                text=doc.page_content or "",
                source=str(meta.get("source", "Instant Case")),
                page=meta.get("page"),
                source_type="record",
                score=score,
            )
            if len(evidence) > before:
                instant_hits += 1

    # Cap so the reason prompt stays readable. Prefer local notes + selection.
    local_types = ("notebook", "user_note", "annotation")
    local = [e for e in evidence if e.get("source_type") in local_types]
    corpus = [e for e in evidence if e.get("source_type") not in local_types]
    # Keep Instant Case / record passages when present.
    record = [e for e in corpus if e.get("source_type") == "record"]
    other = [e for e in corpus if e.get("source_type") != "record"]
    keep_local = local[:3]
    # Instant Case hard questions need record room; do not let Youngstown crowd them out.
    if instant_hits:
        keep_record = record[:9]
        keep_other = other[:2]
    else:
        keep_record = record[:3]
        keep_other = other[: max(0, 10 - len(keep_local) - len(keep_record))]
    evidence = keep_local + keep_record + keep_other
    for i, hit in enumerate(evidence):
        hit["id"] = f"ev-{i}"

    notes = (
        f"Retrieved {len(evidence)} passage(s) from all uploaded PDFs "
        "(Instant Case, Case library, and Articles)."
    )
    if notes_added:
        notes += f" Included {notes_added} local note/annotation chunk(s) from this browser."
    if selection:
        notes += " Selected highlight is early evidence and drove the retrieval query."
    if source_file:
        notes += f" Preferred source article: {source_file}"
        if source_page is not None:
            notes += f" p.{source_page}"
        notes += f" ({same_source_hits} nearby chunk(s))."
    if instant_hits:
        notes += f" Added {instant_hits} Instant Case / record chunk(s)."

    return {
        "evidence": evidence,
        "grounding_notes": notes,
    }
