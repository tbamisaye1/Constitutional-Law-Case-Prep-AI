"""
Verify node: check quotes in the model reply against retrieved evidence.

Runs after reason. Does not call an LLM. Updates grounding_status so the
API / UI can show grounded | partial | abstained | unverified.
"""

from langchain_core.messages import AIMessage

from app.agents.state import PrepState
from app.grounding.verify import extract_simple_claims_from_reply, verify_claims


def verify_claims_node(state: PrepState) -> dict:
    # If we already abstained, do not overwrite that status.
    if state.get("grounding_status") == "abstained":
        return {}

    evidence = list(state.get("evidence") or [])
    last = list(state["messages"])[-1] if state.get("messages") else None
    reply = ""
    if isinstance(last, AIMessage) or getattr(last, "type", "") == "ai":
        content = getattr(last, "content", "")
        reply = content if isinstance(content, str) else str(content)

    claims = extract_simple_claims_from_reply(reply, evidence)
    report = verify_claims(claims, evidence)

    # No extractable quotes: still mark partial so the UI warns you to check
    # the PDFs yourself before using the text in OA.
    status = report["status"]
    if not claims and evidence and status == "unverified":
        notes = (
            "Reply had no clear quotes to auto-verify. Treat as draft; "
            "open the cited PDFs before using in argument."
        )
    else:
        notes = report["notes"]

    return {
        "claims": report["claims"],
        "grounding_status": status,
        "grounding_notes": notes,
    }
