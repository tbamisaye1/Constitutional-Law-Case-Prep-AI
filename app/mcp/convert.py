"""
Markdown ↔ TipTap-safe HTML for MCP note tools.

Reads return both formats. Writes accept exactly one of markdown or html.
Only the TipTap-safe subset survives conversion: h2, h3, p, ul, ol, li,
strong, em, a, blockquote, code, plus Arguments sub-point headings
(<h1 data-outline="point" data-id="…">), which live inside a prong's notes.
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


_POINT_RE = re.compile(r"<h1\b([^>]*)>(.*?)</h1>", re.I | re.S)
_POINT_ATTR_RE = re.compile(r'data-outline\s*=\s*["\']point["\']', re.I)
_POINT_ID_RE = re.compile(r'data-id\s*=\s*["\']([A-Za-z0-9_-]{1,80})["\']', re.I)
_POINT_OPEN = "\x00POINT:{}\x00"
_POINT_CLOSE = "\x00/POINT\x00"


def _protect_points(html: str) -> str:
    """
    Swap sub-point headings for sentinels so the tag pass cannot strip them.

    The Arguments page stores 1.1.1 sub-points as h1 headings inside a prong's
    notes. Any other h1 is still dropped (only its text survives).
    """

    def _swap(match: re.Match[str]) -> str:
        attrs, inner = match.group(1), match.group(2)
        if not _POINT_ATTR_RE.search(attrs):
            return match.group(0)
        id_match = _POINT_ID_RE.search(attrs)
        point_id = id_match.group(1) if id_match else ""
        return _POINT_OPEN.format(point_id) + inner + _POINT_CLOSE

    return _POINT_RE.sub(_swap, html)


def _restore_points(html: str) -> str:
    def _open(match: re.Match[str]) -> str:
        point_id = match.group(1)
        id_attr = f' data-id="{point_id}"' if point_id else ""
        return f'<h1 data-outline="point"{id_attr}>'

    html = re.sub(r"\x00POINT:([A-Za-z0-9_-]*)\x00", _open, html)
    return html.replace(_POINT_CLOSE, "</h1>")


def sanitize_tiptap_html(html: str) -> str:
    """Strip tags outside the TipTap-safe subset; keep href on anchors."""
    if not html:
        return ""
    # Sentinels are NUL-delimited; never trust any that arrive from outside.
    cleaned = _SCRIPT_RE.sub("", html.replace("\x00", ""))
    cleaned = _protect_points(cleaned)

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

    return _restore_points(_TAG_RE.sub(_replace, cleaned))


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
