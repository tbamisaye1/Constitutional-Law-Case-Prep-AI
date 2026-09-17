"""
Web-grounded chat through OpenAI or OpenRouter.

Used only in Ask AI "Web" mode (corpus + web). Documents-only mode never
imports this. A direct OpenAI key uses the Responses API `web_search` tool.
OpenRouter remains the fallback. Both return URLs as EvidenceHit rows.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import get_settings
from app.grounding.schemas import EvidenceHit


def _format_corpus_block(evidence: list[EvidenceHit]) -> str:
    if not evidence:
        return "(no uploaded-article passages retrieved)"
    lines: list[str] = []
    for e in evidence:
        page = f" p.{e['page']}" if e.get("page") else ""
        lines.append(
            f"[{e['id']}] ({e.get('source_type', 'unknown')}) {e['source']}{page}\n{e['text']}\n"
        )
    return "\n".join(lines)


def _citation_to_hit(index: int, annotation: dict[str, Any]) -> EvidenceHit | None:
    """Normalize OpenRouter url_citation (or flat) annotations into EvidenceHit."""
    payload = annotation.get("url_citation") or annotation
    url = (payload.get("url") or "").strip()
    if not url:
        return None
    title = (payload.get("title") or url).strip()
    snippet = (
        payload.get("content")
        or payload.get("text")
        or payload.get("snippet")
        or title
    )
    return {
        "id": f"web-{index}",
        "text": str(snippet).strip(),
        "source": title if title != url else url,
        "page": None,
        "source_type": "web",
        "score": None,
        "url": url,
    }


def extract_web_evidence(message: dict[str, Any]) -> list[EvidenceHit]:
    """Pull web citations from the assistant message annotations / citations."""
    hits: list[EvidenceHit] = []
    seen: set[str] = set()

    def add(hit: EvidenceHit | None) -> None:
        if not hit:
            return
        key = hit.get("url") or hit["source"]
        if key in seen:
            return
        seen.add(key)
        hits.append(hit)

    annotations = message.get("annotations") or []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        kind = ann.get("type") or ""
        if kind in ("url_citation", "citation") or "url" in ann or "url_citation" in ann:
            add(_citation_to_hit(len(hits), ann))

    # Some providers nest citations under message.citations
    for cite in message.get("citations") or []:
        if isinstance(cite, dict):
            add(_citation_to_hit(len(hits), cite))
        elif isinstance(cite, str) and cite.startswith("http"):
            add(
                {
                    "id": f"web-{len(hits)}",
                    "text": cite,
                    "source": cite,
                    "page": None,
                    "source_type": "web",
                    "score": None,
                    "url": cite,
                }
            )

    return hits


def extract_openai_response(data: dict[str, Any]) -> tuple[str, list[EvidenceHit]]:
    """Read text and clickable URL citations from an OpenAI Responses payload."""
    text_parts: list[str] = []
    hits: list[EvidenceHit] = []
    seen: set[str] = set()

    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "output_text":
                continue
            text = block.get("text") or ""
            if text:
                text_parts.append(str(text))
            for annotation in block.get("annotations") or []:
                if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                    continue
                hit = _citation_to_hit(len(hits), annotation)
                if not hit:
                    continue
                key = hit.get("url") or hit["source"]
                if key in seen:
                    continue
                seen.add(key)
                hits.append(hit)

    return "\n".join(text_parts).strip(), hits


def _build_user_content(question: str, corpus_evidence: list[EvidenceHit]) -> str:
    corpus_block = _format_corpus_block(corpus_evidence)
    return (
        "UPLOADED ARTICLES (prefer these when they answer the question):\n"
        f"{corpus_block}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "If the uploaded passages are enough, answer from them and cite [ev-N]. "
        "If you need outside definitions, background, news, or the corpus is thin, "
        "use web search. Cite web sources with title + URL. "
        "Do not invent facts that appear in neither the uploaded passages nor search results."
    )


def _chat_with_openai(
    question: str,
    corpus_evidence: list[EvidenceHit],
    system_prompt: str,
) -> tuple[str, list[EvidenceHit], str]:
    """Use OpenAI's Responses API with its hosted web_search tool."""
    s = get_settings()
    body: dict[str, Any] = {
        "model": s.openai_web_model,
        "tools": [{"type": "web_search", "search_context_size": "medium"}],
        "tool_choice": "auto",
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_content(question, corpus_evidence)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {s.openai_api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=90.0) as client:
        res = client.post("https://api.openai.com/v1/responses", headers=headers, json=body)
        if res.status_code >= 400:
            raise RuntimeError(f"OpenAI web search failed ({res.status_code}): {res.text[:800]}")
        data = res.json()

    reply, web_hits = extract_openai_response(data)
    notes = (
        f"Web mode (OpenAI): {len(corpus_evidence)} corpus hit(s), "
        f"{len(web_hits)} web cite(s)."
    )
    return reply, web_hits, notes


def _chat_with_openrouter(
    question: str,
    corpus_evidence: list[EvidenceHit],
    system_prompt: str,
) -> tuple[str, list[EvidenceHit], str]:
    """Fallback: OpenRouter Chat Completions plus openrouter:web_search."""
    s = get_settings()
    user_content = _build_user_content(question, corpus_evidence)

    body: dict[str, Any] = {
        "model": s.openrouter_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "tools": [
            {
                "type": "openrouter:web_search",
                "parameters": {
                    "engine": "auto",
                    "max_results": 5,
                    "max_total_results": 10,
                },
            }
        ],
        "max_tool_calls": 4,
    }

    url = f"{s.openrouter_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {s.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tbamisaye1/Constitutional-Law-Case-Prep-AI",
        "X-Title": "Case Prep Ask AI Web",
    }

    with httpx.Client(timeout=90.0) as client:
        res = client.post(url, headers=headers, json=body)
        if res.status_code >= 400:
            detail = res.text[:800]
            raise RuntimeError(f"OpenRouter web chat failed ({res.status_code}): {detail}")
        data = res.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(data)[:400]}")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)

    reply = content if isinstance(content, str) else str(content)
    web_hits = extract_web_evidence(message)
    usage = data.get("usage") or {}
    tool_use = usage.get("server_tool_use") or {}
    searches = tool_use.get("web_search_requests")
    note_bits = [
        f"Web mode (OpenRouter): {len(corpus_evidence)} corpus hit(s), "
        f"{len(web_hits)} web cite(s)."
    ]
    if searches is not None:
        note_bits.append(f"OpenRouter ran {searches} web search call(s).")
    return reply.strip(), web_hits, " ".join(note_bits)


def chat_with_web_search(
    question: str,
    corpus_evidence: list[EvidenceHit],
    *,
    system_prompt: str,
) -> tuple[str, list[EvidenceHit], str]:
    """
    Prefer direct OpenAI web search when OPENAI_API_KEY exists.

    Returns (reply_text, web_evidence_hits, notes).
    """
    s = get_settings()
    if s.openai_api_key:
        return _chat_with_openai(question, corpus_evidence, system_prompt)
    if s.openrouter_api_key:
        return _chat_with_openrouter(question, corpus_evidence, system_prompt)
    raise RuntimeError("OPENAI_API_KEY and OPENROUTER_API_KEY are both missing")
