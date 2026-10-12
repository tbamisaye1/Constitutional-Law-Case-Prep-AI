from app.agents.nodes.retrieve import _prefer_uploads_over_oyez


def test_drops_oyez_when_matching_pdf_upload_is_present():
    hits = [
        {"source": "Ex parte Milligan (Oyez summary)", "text": "oyez"},
        {"source": "Ex parte Milligan, 71 U.S. (4 Wall.) 2.pdf", "text": "pdf"},
        {"source": "Katz v. United States (Oyez summary)", "text": "katz oyez"},
    ]
    out = _prefer_uploads_over_oyez(hits)
    sources = [h["source"] for h in out]
    assert "Ex parte Milligan (Oyez summary)" not in sources
    assert "Ex parte Milligan, 71 U.S. (4 Wall.) 2.pdf" in sources
    # No Katz upload in the hit list — keep the Oyez row.
    assert "Katz v. United States (Oyez summary)" in sources


def test_keeps_oyez_when_no_upload_overlap():
    hits = [
        {"source": "Ex parte Milligan (Oyez summary)", "text": "oyez"},
        {"source": "Youngstown Sheet & Tube Co. v. Sawyer.pdf", "text": "pdf"},
    ]
    out = _prefer_uploads_over_oyez(hits)
    sources = [h["source"] for h in out]
    assert "Ex parte Milligan (Oyez summary)" in sources
    assert "Youngstown Sheet & Tube Co. v. Sawyer.pdf" in sources
