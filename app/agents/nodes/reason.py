"""
Reason node: answer only from retrieved evidence, or abstain.

Anti-hallucination design choice: we check abstain *before* calling the
model. If the corpus is empty, we never give the LLM a chance to invent
Carpenter holdings from parametric memory.

Web mode (grounding_source=web_plus) is different: corpus still runs first,
then OpenRouter web_search may fill gaps. Documents mode is unchanged.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import PrepState
from app.grounding.abstain import make_abstain_message, should_abstain
from app.grounding.prompts import (
    GROUNDED_LEGAL_SYSTEM,
    WEB_ABSTAIN_TEMPLATE,
    WEB_PLUS_SYSTEM,
    build_reason_user_payload,
)
from app.grounding.schemas import EvidenceHit
from app.llm.openrouter import get_chat_model


def _format_evidence(evidence: list[EvidenceHit]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for e in evidence:
        page = f" p.{e['page']}" if e.get("page") else ""
        url = e.get("url")
        loc = f" {url}" if url else page
        lines.append(
            f"[{e['id']}] ({e['source_type']}) {e['source']}{loc}\n{e['text']}\n"
        )
    return "\n".join(lines)


def _latest_user_text(state: PrepState) -> str:
    for msg in reversed(list(state["messages"])):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _reason_documents(state: PrepState) -> dict:
    """Existing RAG-only path. Leave behavior alone."""
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
    return {
        "messages": [response],
        "grounding_status": "unverified",
    }


def _reason_web_plus(state: PrepState) -> dict:
    """
    Corpus + OpenRouter web search.

    Does not abstain solely because FAISS is thin. The model may search the
    web (e.g. "what is a radiological dirty bomb") while still seeing PDF hits.
    """
    from app.config import get_settings
    from app.llm.web_chat import chat_with_web_search

    evidence = list(state.get("evidence") or [])
    question = _latest_user_text(state).strip()
    if not question:
        return {
            "messages": [AIMessage(content=WEB_ABSTAIN_TEMPLATE.format(reason="Empty question."))],
            "grounding_status": "abstained",
            "claims": [],
            "grounding_notes": "Empty question.",
        }

    if not get_settings().web_search_configured:
        return {
            "messages": [
                AIMessage(
                    content=WEB_ABSTAIN_TEMPLATE.format(
                        reason=(
                            "OPENAI_API_KEY and OPENROUTER_API_KEY are both missing, "
                            "so web search cannot run."
                        )
                    )
                )
            ],
            "grounding_status": "abstained",
            "claims": [],
            "grounding_notes": "Missing web-search API key.",
        }

    try:
        reply, web_hits, notes = chat_with_web_search(
            question,
            evidence,
            system_prompt=WEB_PLUS_SYSTEM,
        )
    except Exception as exc:  # noqa: BLE001 — surface API failures as abstain, not 500
        return {
            "messages": [AIMessage(content=WEB_ABSTAIN_TEMPLATE.format(reason=str(exc)))],
            "grounding_status": "abstained",
            "claims": [],
            "grounding_notes": f"Web search call failed: {exc}",
            "evidence": evidence,
        }

    merged = list(evidence) + list(web_hits)
    if not reply.strip():
        return {
            "messages": [
                AIMessage(
                    content=WEB_ABSTAIN_TEMPLATE.format(
                        reason="The model returned an empty reply after web search."
                    )
                )
            ],
            "grounding_status": "abstained",
            "claims": [],
            "grounding_notes": notes,
            "evidence": merged,
        }

    return {
        "messages": [AIMessage(content=reply)],
        "grounding_status": "unverified",
        "grounding_notes": notes,
        "evidence": merged,
    }


def reason_node(state: PrepState) -> dict:
    mode = (state.get("grounding_source") or "documents").strip().lower()
    if mode == "web_plus":
        return _reason_web_plus(state)
    return _reason_documents(state)
