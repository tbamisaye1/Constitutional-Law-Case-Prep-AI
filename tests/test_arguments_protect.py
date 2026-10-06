from app.db.repository import (
    _merge_arguments_boards,
    _pick_argument_notes,
    _protect_arguments_from_seed_or_thinner_push,
)


class _Cursor:
    def __init__(self, existing):
        self._existing = existing

    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return self._existing


def test_seed_notes_lose_to_manual_even_when_seed_is_longer():
    seedish = "<h2>Hamdi: concede, then confine</h2><ul><li>" + ("x" * 3000) + "</li></ul>"
    manual = "<p><strong>Look at what Congress woul dhave inteded for his</strong></p>"
    # Manual is shorter but not seed — must win when incoming is seed-shaped.
    assert "Look at what Congress" in _pick_argument_notes(manual, seedish, "c3-s2-a")


def test_merge_keeps_manual_jackson_over_seed_push():
    server = {
        "draftsBySide": {
            "petitioner": [
                {
                    "id": "alt-q2-ladder",
                    "sections": [
                        {
                            "id": "c3-s1",
                            "prongs": [
                                {
                                    "id": "c3-s1-a",
                                    "title": "a. Jackson’s method, not just his labels",
                                    "notes": "<h2>The claim in one sentence</h2><p>Where Congress has legislated</p>",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }
    incoming = {
        "draftsBySide": {
            "petitioner": [
                {
                    "id": "alt-q2-ladder",
                    "sections": [
                        {
                            "id": "c3-s1",
                            "prongs": [
                                {
                                    "id": "c3-s1-a",
                                    "title": "a. Jackson’s method, not just his labels",
                                    "notes": "<h2>Walk the three steps he walked</h2><ol><li><p>Category 1 out: the government</p></li></ol>",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }
    merged = _merge_arguments_boards(incoming, server)
    notes = merged["draftsBySide"]["petitioner"][0]["sections"][0]["prongs"][0]["notes"]
    assert "The claim in one sentence" in notes


def test_protect_arguments_uses_server_when_push_is_seed():
    existing = {
        "data": {
            "draftsBySide": {
                "petitioner": [
                    {
                        "id": "alt-q2-ladder",
                        "sections": [
                            {
                                "id": "c3-s1",
                                "prongs": [
                                    {
                                        "id": "c3-s1-a",
                                        "title": "a. Under Youngstown",
                                        "notes": "<h2>The claim in one sentence</h2><p>manual</p>",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        }
    }
    prepared = {
        "kind": "arguments",
        "id": "main",
        "data": {
            "draftsBySide": {
                "petitioner": [
                    {
                        "id": "alt-q2-ladder",
                        "sections": [
                            {
                                "id": "c3-s1",
                                "prongs": [
                                    {
                                        "id": "c3-s1-a",
                                        "title": "a. Jackson’s method, not just his labels",
                                        "notes": "<h2>Walk the three steps he walked</h2><ol><li>Category 1 out</li></ol>",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        },
    }
    out = _protect_arguments_from_seed_or_thinner_push(_Cursor(existing), "ws", prepared)
    notes = out["data"]["draftsBySide"]["petitioner"][0]["sections"][0]["prongs"][0]["notes"]
    assert "The claim in one sentence" in notes
