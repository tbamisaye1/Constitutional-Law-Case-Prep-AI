"""
Reason node: answer only from retrieved evidence, or abstain.

Anti-hallucination design choice: we check abstain *before* calling the
model. If the corpus is empty, we never give the LLM a chance to invent
Carpenter holdings from parametric memory.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import PrepState
from app.grounding.abstain import make_abstain_message, should_abstain
from app.grounding.prompts import GROUNDED_LEGAL_SYSTEM, build_reason_user_payload
from app.grounding.schemas import EvidenceHit
from app.llm.openrouter import get_chat_model


def _format_evidence(evidence: list[EvidenceHit]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for e in evidence:
        page = f" p.{e['page']}" if e.get("page") else ""
        lines.append(
            f"[{e['id']}] ({e['source_type']}) {e['source']}{page}\n{e['text']}\n"
        )
    return "\n".join(lines)


def _latest_user_text(state: PrepState) -> str:
    for msg in reversed(list(state["messages"])):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def reason_node(state: PrepState) -> dict:
    from app.config import get_settings

    evidence = list(state.get("evidence") or [])
    abstain, reason = should_abstain(evidence)
    if abstain:
        text = make_abstain_message(reason)
        return {
            "messages": [AIMessage(content=text)],
            "grounding_status": "abstained",
            "claims": [],
            "grounding_notes": reason,
        }

    if not get_settings().openrouter_api_key:
        msg = make_abstain_message(
            "Evidence was retrieved, but OPENROUTER_API_KEY is missing so we will not call a model."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "grounding_status": "abstained",
            "claims": [],
            "grounding_notes": "Missing OpenRouter API key.",
        }

    question = _latest_user_text(state)
    payload = build_reason_user_payload(question, _format_evidence(evidence))
    model = get_chat_model()
    response = model.invoke(
        [
            SystemMessage(content=GROUNDED_LEGAL_SYSTEM),
            HumanMessage(content=payload),
        ]
    )
    # Status still "unverified" until verify_claims runs.
    return {
        "messages": [response],
        "grounding_status": "unverified",
    }
