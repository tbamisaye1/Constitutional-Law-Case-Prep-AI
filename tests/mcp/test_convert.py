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


def test_sanitize_keeps_argument_sub_points():
    dirty = (
        '<h1 data-outline="point" data-id="pt-1-abc" class="outline-heading is-point">'
        "Why <strong>Category 3</strong></h1><p>notes</p><h1>plain title</h1>"
    )
    clean = sanitize_tiptap_html(dirty)
    assert '<h1 data-outline="point" data-id="pt-1-abc">Why <strong>Category 3</strong></h1>' in clean
    assert "class=" not in clean
    assert "<h1>" not in clean  # a bare h1 is still dropped
    assert "plain title" in clean


def test_sanitize_drops_unsafe_point_ids_and_smuggled_sentinels():
    clean = sanitize_tiptap_html(
        '<h1 data-outline="point" data-id="x\" onclick=1">t</h1>\x00POINT:evil\x00<p>x</p>'
    )
    assert "onclick" not in clean
    assert '<h1 data-outline="point" data-id="x">t</h1>' in clean  # id cut at the quote
    assert "evil" in clean and "\x00" not in clean and clean.count("<h1") == 1
