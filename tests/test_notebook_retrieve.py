"""Unit tests for notebook evidence merge in retrieve."""

from langchain_core.messages import HumanMessage

from app.agents.nodes.retrieve import _merge_client_notes, retrieve_node
from app.grounding.abstain import should_abstain


def test_merge_client_notes_adds_notebook_hits():
    evidence = []
    added = _merge_client_notes(
        evidence,
        [
            {
                "title": "NDAA detention",
                "text": "The 2012 NDAA affirms AUMF detention authority for persons captured in hostilities.",
                "section_name": "Issue 2 Notes",
                "notes_path": "/notes?section=sec-issue2&page=pg-ndaa",
            }
        ],
    )
    assert added == 1
    assert evidence[0]["source_type"] == "notebook"
    assert evidence[0]["notes_path"].startswith("/notes?")
    assert "NDAA" in evidence[0]["text"]
    assert "Notes" in evidence[0]["source"]


def test_merge_client_notes_adds_annotation_hits():
    evidence = []
    added = _merge_client_notes(
        evidence,
        [
            {
                "title": "Highlight · p.3",
                "text": 'Quote: “enemy combatant”\nCourt requires notice.',
                "section_name": "Hamdi v. Rumsfeld · Hamdi.pdf",
                "notes_path": "/library?case=hamdi&file=pdf-1&page=3",
                "source_type": "annotation",
                "page": 3,
            }
        ],
    )
    assert added == 1
    assert evidence[0]["source_type"] == "annotation"
    assert evidence[0]["page"] == 3
    assert evidence[0]["notes_path"].startswith("/library?")
    assert "Annotation" in evidence[0]["source"]


def test_should_not_abstain_on_notebook_only():
    evidence = [
        {
            "id": "ev-0",
            "text": "x" * 100,
            "source": "NDAA (Notes)",
            "page": None,
            "source_type": "notebook",
            "score": 0.0,
            "notes_path": "/notes?section=a&page=b",
        }
    ]
    abstain, _ = should_abstain(evidence)
    assert abstain is False


def test_retrieve_with_notes_when_index_empty(monkeypatch):
    monkeypatch.setattr("app.agents.nodes.retrieve.load_store", lambda: None)
    state = {
        "messages": [HumanMessage(content="What did I write about NDAA?")],
        "client_notes": [
            {
                "title": "NDAA",
                "text": "The 2012 NDAA affirms AUMF detention authority " + ("more text " * 20),
                "section_name": "Issue 2",
                "notes_path": "/notes?section=sec&page=pg",
            }
        ],
    }
    out = retrieve_node(state)
    assert len(out["evidence"]) >= 1
    assert out["evidence"][0]["source_type"] == "notebook"
    assert "note" in out["grounding_notes"].lower()
