"""
Abstain rules: when the safe answer is "I don't know yet."

Legal products fail when they fill gaps with training-data guesses.
These helpers decide before/after generation whether we should refuse.
"""

from app.grounding.prompts import ABSTAIN_TEMPLATE
from app.grounding.schemas import EvidenceHit


# Minimum characters of retrieved text before we allow a grounded answer.
MIN_EVIDENCE_CHARS = 80

# FAISS similarity_search_with_score usually returns L2 *distance* (lower is
# better). We abstain only if every hit is farther than this. Tune after you
# index a few Bronner PDFs; until then character-length checks do most work.
MAX_L2_DISTANCE = 1.8


def should_abstain(evidence: list[EvidenceHit]) -> tuple[bool, str]:
    """
    Return (True, reason) if we must not let the model invent an answer.
    """
    if not evidence:
        return True, "No chunks were retrieved. The vector store may be empty."

    total_chars = sum(len(e["text"]) for e in evidence)
    if total_chars < MIN_EVIDENCE_CHARS:
        return True, "Retrieved text is too short to support a legal claim."

    scored = [e for e in evidence if e.get("score") is not None]
    if scored and all((e["score"] or 999) > MAX_L2_DISTANCE for e in scored):
        return True, "Retrieval distances are high; the question may be outside the corpus."

    return False, ""


def make_abstain_message(reason: str) -> str:
    return ABSTAIN_TEMPLATE.format(reason=reason)
