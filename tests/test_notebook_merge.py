from app.db.notebook_merge import merge_notebook_snapshots


def test_keeps_server_section_with_pages_when_client_wiped_it():
    remote_incoming = {
        "tree": [
            {
                "id": "grp-bronner",
                "name": "Bronner",
                "kind": "group",
                "children": [
                    {"id": "sec-issue1", "name": "Issue 1", "kind": "section"},
                ],
            }
        ],
        "pagesBySection": {
            "sec-issue1": [{"id": "pg-1", "title": "A", "html": "<p>x</p>"}],
        },
    }
    server_existing = {
        "tree": [
            {
                "id": "grp-bronner",
                "name": "Bronner",
                "kind": "group",
                "children": [
                    {"id": "sec-issue1", "name": "Issue 1", "kind": "section"},
                    {"id": "sec-bg", "name": "Background info", "kind": "section"},
                ],
            }
        ],
        "pagesBySection": {
            "sec-issue1": [{"id": "pg-1", "title": "A", "html": "<p>x</p>"}],
            "sec-bg": [{"id": "pg-ndaa", "title": "NDAA", "html": "<p>covered</p>"}],
        },
    }

    merged = merge_notebook_snapshots(remote_incoming, server_existing)
    kids = merged["tree"][0]["children"]
    assert any(c["id"] == "sec-bg" for c in kids)
    assert merged["pagesBySection"]["sec-bg"][0]["html"] == "<p>covered</p>"


def test_drops_empty_secondary_only_stubs():
    primary = {
        "tree": [{"id": "sec-1", "name": "Keep", "kind": "section"}],
        "pagesBySection": {"sec-1": []},
    }
    secondary = {
        "tree": [
            {"id": "sec-1", "name": "Keep", "kind": "section"},
            {"id": "sec-empty", "name": "New section", "kind": "section"},
        ],
        "pagesBySection": {"sec-1": [], "sec-empty": []},
    }
    merged = merge_notebook_snapshots(primary, secondary)
    assert [n["id"] for n in merged["tree"]] == ["sec-1"]
