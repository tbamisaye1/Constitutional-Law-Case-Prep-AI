"""Markdown ↔ TipTap HTML round trips."""

from __future__ import annotations

from app.mcp.convert import html_to_markdown, markdown_to_html, sanitize_tiptap_html


def test_markdown_round_trip_preserves_structure():
    html = (
        "<h2>Holding</h2><p>The Court held that <strong>Article II</strong> "
        "does not authorize this.</p><ul><li>First</li><li><em>Second</em></li></ul>"
    )
    md = html_to_markdown(html)
    back = markdown_to_html(md)
    assert "<h2>" in back
    assert "Article II" in back
    assert "<strong>" in back or "<b>" in back
    assert "<ul>" in back
    assert "<li>" in back


def test_sanitize_strips_script():
    dirty = '<p>ok</p><script>alert(1)</script><h1>no</h1>'
    clean = sanitize_tiptap_html(dirty)
    assert "script" not in clean.lower()
    assert "<h1>" not in clean
    assert "<p>" in clean
