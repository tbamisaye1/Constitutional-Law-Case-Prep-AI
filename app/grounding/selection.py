"""
Pack / unpack a highlighted PDF passage with the user's question.

Ask AI used to send only the short question ("what does this mean"). FAISS then
retrieved unrelated Bronner / Fourth Amendment chunks and the model answered
those. The selected text has to travel with the question and drive retrieval.
"""

from __future__ import annotations

import re

_SELECTION_BLOCK = re.compile(
    r"SELECTED PASSAGE \(explain this; do not ignore it\):\s*\"\"\"\s*(.*?)\s*\"\"\"",
    re.DOTALL | re.IGNORECASE,
)
_USER_QUESTION = re.compile(
    r"USER QUESTION:\s*(.*)\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def build_chat_message(question: str, selection: str | None = None) -> str:
    """Compose the HumanMessage content the agent graph sees."""
    q = (question or "").strip()
    sel = (selection or "").strip()
    if not sel:
        return q
    return (
        "SELECTED PASSAGE (explain this; do not ignore it):\n"
        f'"""\n{sel}\n"""\n\n'
        f"USER QUESTION:\n{q}"
    )


def extract_selection(message: str) -> str:
    """Return the highlighted passage if the message was packed that way."""
    match = _SELECTION_BLOCK.search(message or "")
    if not match:
        return ""
    return match.group(1).strip()


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
    """
    selection = extract_selection(message)
    question = extract_user_question(message)
    if selection and question:
        # Passage first so Padilla / AUMF tokens dominate the embedding.
        return f"{selection}\n\n{question}"
    return selection or question or (message or "").strip()
