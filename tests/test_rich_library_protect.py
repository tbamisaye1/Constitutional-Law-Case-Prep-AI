from app.db.repository import _protect_rich_library_text_from_thinner_push


class _Cursor:
    def __init__(self, existing):
        self._existing = existing

    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return self._existing


def test_keeps_longer_opinion_body_when_push_is_thinner():
    existing = {
        "data": {
            "id": "op-hamdi-maj",
            "caseId": "hamdi",
            "bodyHtml": "<p>" + ("x" * 500) + "</p>",
            "notes": "<p>" + ("x" * 500) + "</p>",
            "summary": "held",
        }
    }
    prepared = {
        "kind": "opinions",
        "id": "op-hamdi-maj",
        "data": {
            "id": "op-hamdi-maj",
            "caseId": "hamdi",
            "bodyHtml": "<p>short</p>",
            "notes": "",
            "summary": "held",
        },
    }
    out = _protect_rich_library_text_from_thinner_push(_Cursor(existing), "ws", prepared)
    assert len(out["data"]["bodyHtml"]) > 100
    assert out["data"]["notes"] == out["data"]["bodyHtml"]
