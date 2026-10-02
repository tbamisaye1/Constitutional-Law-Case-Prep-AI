"""
Pack / unpack a highlighted PDF passage with the user's question.

Ask AI used to send only the short question ("what does this mean"). FAISS then
retrieved unrelated Bronner / Fourth Amendment chunks and the model answered
those. The selected text has to travel with the question and drive retrieval.

When the highlight comes from a known uploaded article, also pack the source
filename + page so the model and retriever know which PDF to prefer.
"""

from __future__ import annotations

import re

_SELECTION_BLOCK = re.compile(
    r"SELECTED PASSAGE \(explain this; do not ignore it\):\s*\"\"\"\s*(.*?)\s*\"\"\"",
    re.DOTALL | re.IGNORECASE,
)
_SOURCE_BLOCK = re.compile(
    r"SOURCE ARTICLE:\s*(.+?)(?:\s*\(p\.\s*(\d+)\))?\s*(?:\n|$)",
    re.IGNORECASE,
)
_USER_QUESTION = re.compile(
    r"USER QUESTION:\s*(.*)\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def build_chat_message(
    question: str,
    selection: str | None = None,
    *,
    source_file: str | None = None,
    page: int | None = None,
) -> str:
    """Compose the HumanMessage content the agent graph sees."""
    q = (question or "").strip()
    sel = (selection or "").strip()
    source = (source_file or "").strip()
    if not sel:
        return q

    parts: list[str] = [
        "SELECTED PASSAGE (explain this; do not ignore it):",
        f'"""\n{sel}\n"""',
    ]
    if source:
        if page is not None:
            parts.append(f"SOURCE ARTICLE: {source} (p. {int(page)})")
        else:
            parts.append(f"SOURCE ARTICLE: {source}")
        parts.append(
            "Prefer this uploaded article for context. The user may be reading "
            "to understand the article, not arguing petitioner/respondent."
        )
    parts.append(f"USER QUESTION:\n{q}")
    return "\n\n".join(parts)


def extract_selection(message: str) -> str:
    """Return the highlighted passage if the message was packed that way."""
    match = _SELECTION_BLOCK.search(message or "")
    if not match:
        return ""
    return match.group(1).strip()


def extract_source_file(message: str) -> str:
    """Return the SOURCE ARTICLE filename if packed into the message."""
    match = _SOURCE_BLOCK.search(message or "")
    if not match:
        return ""
    return (match.group(1) or "").strip()


def extract_source_page(message: str) -> int | None:
    """Return the SOURCE ARTICLE page if present."""
    match = _SOURCE_BLOCK.search(message or "")
    if not match or not match.group(2):
        return None
    try:
        return int(match.group(2))
    except ValueError:
        return None


def extract_user_question(message: str) -> str:
    """Return the user's short question, or the whole message if unscoped."""
    text = message or ""
    if not extract_selection(text):
        return text.strip()
    match = _USER_QUESTION.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def retrieval_query(message: str) -> str:
    """
    Embed this for FAISS.

    Prefer the selected passage (specific case names, holdings) over a vague
    question like "what does this mean", which otherwise matches random corpus.
    Include the source filename when known so same-PDF chunks rank closer.
    """
    selection = extract_selection(message)
    question = extract_user_question(message)
    source = extract_source_file(message)
    bits: list[str] = []
    if selection:
        bits.append(selection)
    if question:
        bits.append(question)
    if source:
        bits.append(source)
    if bits:
        return "\n\n".join(bits)
    return (message or "").strip()
