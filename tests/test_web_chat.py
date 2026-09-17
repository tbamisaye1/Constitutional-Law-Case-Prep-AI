"""Unit tests for OpenRouter web citation parsing (no network)."""

from app.llm.web_chat import extract_openai_response, extract_web_evidence


def test_extract_url_citation_annotations():
    message = {
        "content": "A radiological dirty bomb spreads radioactive material.",
        "annotations": [
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.com/dirty-bomb",
                    "title": "Dirty bomb FAQ",
                    "content": "A dirty bomb combines conventional explosives with radioactive material.",
                },
            }
        ],
    }
    hits = extract_web_evidence(message)
    assert len(hits) == 1
    assert hits[0]["source_type"] == "web"
    assert hits[0]["url"] == "https://example.com/dirty-bomb"
    assert "radioactive" in hits[0]["text"]
    assert hits[0]["id"] == "web-0"


def test_extract_dedupes_same_url():
    message = {
        "content": "…",
        "annotations": [
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.com/a",
                    "title": "A",
                    "content": "one",
                },
            },
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.com/a",
                    "title": "A again",
                    "content": "two",
                },
            },
        ],
    }
    hits = extract_web_evidence(message)
    assert len(hits) == 1


def test_extract_string_citations():
    message = {
        "content": "…",
        "citations": ["https://cdc.gov/radiation"],
    }
    hits = extract_web_evidence(message)
    assert len(hits) == 1
    assert hits[0]["url"] == "https://cdc.gov/radiation"


def test_extract_openai_responses_text_and_citations():
    data = {
        "output": [
            {"type": "web_search_call", "id": "search-1", "status": "completed"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "A dirty bomb spreads radioactive material.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://www.cdc.gov/example",
                                "title": "CDC dirty bomb FAQ",
                                "start_index": 0,
                                "end_index": 20,
                            }
                        ],
                    }
                ],
            },
        ]
    }

    text, hits = extract_openai_response(data)

    assert text == "A dirty bomb spreads radioactive material."
    assert len(hits) == 1
    assert hits[0]["source"] == "CDC dirty bomb FAQ"
    assert hits[0]["url"] == "https://www.cdc.gov/example"
