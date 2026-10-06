"""
Markdown ↔ TipTap-safe HTML for MCP note tools.

Reads return both formats. Writes accept exactly one of markdown or html.
Only the TipTap-safe subset survives conversion: h2, h3, p, ul, ol, li,
strong, em, a, blockquote, code.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import markdown as md_lib
from markdownify import markdownify as html_to_md

FormatName = Literal["markdown", "html", "both"]

_ALLOWED_TAGS = frozenset(
    {
        "h2",
        "h3",
        "p",
        "ul",
        "ol",
        "li",
        "strong",
        "b",
        "em",
        "i",
        "a",
        "blockquote",
        "code",
        "br",
    }
)

_TAG_RE = re.compile(r"</?([a-zA-Z0-9]+)([^>]*)>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)


def markdown_to_html(text: str) -> str:
    """Convert markdown to TipTap-safe HTML."""
    raw = md_lib.markdown(
        text or "",
        extensions=["extra", "sane_lists", "nl2br"],
        output_format="html",
    )
    return sanitize_tiptap_html(raw)


def html_to_markdown(html: str) -> str:
    """Convert TipTap HTML to markdown for LLM clients."""
    if not html:
        return ""
    return html_to_md(html, heading_style="ATX", bullets="-").strip()


def sanitize_tiptap_html(html: str) -> str:
    """Strip tags outside the TipTap-safe subset; keep href on anchors."""
    if not html:
        return ""
    cleaned = _SCRIPT_RE.sub("", html)

    def _replace(match: re.Match[str]) -> str:
        tag = match.group(1).lower()
        if tag not in _ALLOWED_TAGS:
            return ""
        full = match.group(0)
        if tag == "a" and not full.startswith("</"):
            href_match = re.search(r'href\s*=\s*["\']([^"\']*)["\']', full, re.I)
            href = href_match.group(1) if href_match else ""
            if href.startswith(("http://", "https://", "mailto:", "#")):
                return f'<a href="{href}">'
            return "<a>"
        if full.startswith("</"):
            # Normalize b/i to strong/em for TipTap.
            if tag == "b":
                return "</strong>"
            if tag == "i":
                return "</em>"
            return f"</{tag}>"
        if tag == "b":
            return "<strong>"
        if tag == "i":
            return "<em>"
        if tag == "br":
            return "<br />"
        return f"<{tag}>"

    return _TAG_RE.sub(_replace, cleaned)


def format_note_payload(
    html: str,
    *,
    format: FormatName = "markdown",
) -> dict[str, Any]:
    """Return markdown / html / both for a note body."""
    safe_html = html or ""
    md = html_to_markdown(safe_html)
    if format == "html":
        return {"html": safe_html}
    if format == "both":
        return {"markdown": md, "html": safe_html}
    return {"markdown": md}


def resolve_write_body(
    *,
    markdown: str | None = None,
    html: str | None = None,
) -> str:
    """
    Accept exactly one of markdown or html; return TipTap-safe HTML.

    Raises:
        ValueError: when both or neither are provided.
    """
    has_md = markdown is not None
    has_html = html is not None
    if has_md == has_html:
        raise ValueError("Provide exactly one of markdown or html.")
    if has_md:
        return markdown_to_html(markdown or "")
    return sanitize_tiptap_html(html or "")
