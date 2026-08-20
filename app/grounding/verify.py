"""
Verify that claimed quotes actually appear in retrieved evidence.

This is deterministic string checking, not another LLM call. That matters:
a second model can also hallucinate. Start with overlap; add LLM-as-judge
evals later for softer paraphrase cases.
"""

import re

from app.grounding.schemas import Claim, EvidenceHit, GroundingReport, GroundingStatus


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace so minor PDF quirks do not fail a match."""
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_supported(quote: str, evidence: list[EvidenceHit], support_ids: list[str]) -> bool:
    """
    True if the quote (or a long substring) appears in the supported chunks.
    Short quotes (< 12 chars) are treated as unverified to block fake cites.
    """
    q = normalize(quote)
    if len(q) < 12:
        return False

    allowed = {e["id"]: e for e in evidence}
    pool = []
    for sid in support_ids:
        if sid in allowed:
            pool.append(allowed[sid]["text"])
    if not pool:
        pool = [e["text"] for e in evidence]

    blob = normalize(" ".join(pool))
    if q in blob:
        return True

    # Soft check: first ~40 chars of the quote must appear (handles ellipsis cuts).
    head = q[:40]
    return len(head) >= 12 and head in blob


def verify_claims(claims: list[Claim], evidence: list[EvidenceHit]) -> GroundingReport:
    if not evidence:
        return GroundingReport(
            status="no_evidence",
            claims=claims,
            evidence_ids=[],
            notes="No evidence in state; nothing to verify against.",
        )

    updated: list[Claim] = []
    ok = 0
    for claim in claims:
        verified = quote_supported(claim.get("quote", ""), evidence, claim.get("support_ids", []))
        row: Claim = {
            "claim": claim.get("claim", ""),
            "support_ids": claim.get("support_ids", []),
            "quote": claim.get("quote", ""),
            "verified": verified,
        }
        if verified:
            ok += 1
        updated.append(row)

    status: GroundingStatus
    if not updated:
        status = "unverified"
    elif ok == len(updated):
        status = "grounded"
    elif ok == 0:
        status = "unverified"
    else:
        status = "partial"

    return GroundingReport(
        status=status,
        claims=updated,
        evidence_ids=[e["id"] for e in evidence],
        notes=f"{ok}/{len(updated)} claims passed quote check.",
    )


def extract_simple_claims_from_reply(reply: str, evidence: list[EvidenceHit]) -> list[Claim]:
    """
    Pull lines that look like quoted support from the reply.
    Replace with structured JSON output when the claim schema is wired end-to-end.
    """
    claims: list[Claim] = []
    # Match "..." or “...” snippets in the reply.
    for match in re.finditer(r"[\"“]([^\"”]{12,200})[\"”]", reply):
        quote = match.group(1).strip()
        claims.append(
            {
                "claim": quote[:120],
                "support_ids": [e["id"] for e in evidence[:3]],
                "quote": quote,
                "verified": None,
            }
        )
    return claims
