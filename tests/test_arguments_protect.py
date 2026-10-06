from app.db.repository import (
    _looks_like_seed_draft_notes,
    _merge_arguments_boards,
    _pick_argument_notes,
    _pick_rich_notes,
    _protect_arguments_from_seed_or_thinner_push,
)


class _Cursor:
    def __init__(self, existing):
        self._existing = existing

    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return self._existing


SEED_DRAFT = (
    "<h2>Introduction</h2><p>The Constitution lets the President turn the armed forces "
    "outward against an enemy. We ask this court to reverse for 3 reasons.</p>"
    "<h2>The ladder (say this in the roadmap)</h2><ol><li>Lowest ebb.</li></ol>"
)

MANUAL_DRAFT = (
    "<p>2nd Ebb considerations:</p><p></p><p><strong><em><u>For H</u></em></strong></p>"
    "<p>Longer manual outline that must survive a seed sync from another tab.</p>"
)


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


def test_seed_draft_fingerprint():
    assert _looks_like_seed_draft_notes(SEED_DRAFT) is True
    assert _looks_like_seed_draft_notes(MANUAL_DRAFT) is False


def test_pick_rich_notes_keeps_manual_over_seed():
    assert _pick_rich_notes(MANUAL_DRAFT, SEED_DRAFT) == MANUAL_DRAFT
    assert _pick_rich_notes(SEED_DRAFT, MANUAL_DRAFT) == MANUAL_DRAFT


def test_pick_rich_notes_blocks_short_wipe():
    long_notes = MANUAL_DRAFT + (" more" * 40)
    assert _pick_rich_notes(long_notes, "<p>x</p>") == long_notes


def test_merge_keeps_whole_argument_notes_when_seed_arrives():
    """Regression: draft.notes used to take incoming wholesale and erase 2nd Ebb."""
    server = {
        "draftsBySide": {
            "petitioner": [
                {
                    "id": "cat3",
                    "title": "Category 3",
                    "notes": MANUAL_DRAFT,
                    "sections": [
                        {
                            "id": "c3-s1",
                            "title": "I.",
                            "notes": "",
                            "prongs": [
                                {
                                    "id": "c3-s1-a",
                                    "title": "a. Jackson’s method, not just his labels",
                                    "notes": "<p>claim in one sentence</p>",
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
                    "id": "cat3",
                    "title": "Category 3",
                    "notes": SEED_DRAFT,
                    "sections": [
                        {
                            "id": "c3-s1",
                            "title": "I.",
                            "notes": "",
                            "prongs": [
                                {
                                    "id": "c3-s1-a",
                                    "title": "a. Under Youngstown, President falls into lowest Category",
                                    "notes": "<h2>Walk the three steps</h2><p>seed prong</p>",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }
    merged = _merge_arguments_boards(incoming, server)
    draft = merged["draftsBySide"]["petitioner"][0]
    assert "2nd Ebb" in draft["notes"]
    assert "Introduction" not in draft["notes"]
    prong = draft["sections"][0]["prongs"][0]
    assert "Jackson" in prong["title"]
