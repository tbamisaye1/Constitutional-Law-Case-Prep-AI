from datetime import datetime, timezone

from app.db.repository import (
    _looks_like_seed_draft_notes,
    _merge_arguments_boards,
    _pick_argument_notes,
    _pick_rich_notes,
    _protect_arguments_from_seed_or_thinner_push,
    _protect_arguments_stale_base,
    to_epoch_ms,
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


def test_merge_keeps_manual_jackson_test_over_seed_push():
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
                                    "title": "a. Under Youngstown, President falls into lowest Category",
                                    "notes": "<h2>Jackson's test</h2><p>Category 1 out: the government conceded</p>",
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
                                    "title": "a. Under Youngstown, President falls into lowest Category",
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
    assert "Jackson's test" in notes


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
    out, reason = _protect_arguments_from_seed_or_thinner_push(_Cursor(existing), "ws", prepared)
    notes = out["data"]["draftsBySide"]["petitioner"][0]["sections"][0]["prongs"][0]["notes"]
    assert "The claim in one sentence" in notes
    assert reason is not None


def test_seed_draft_fingerprint():
    assert _looks_like_seed_draft_notes(SEED_DRAFT) is True
    assert _looks_like_seed_draft_notes(MANUAL_DRAFT) is False


def test_pick_rich_notes_keeps_manual_over_seed():
    assert _pick_rich_notes(MANUAL_DRAFT, SEED_DRAFT) == MANUAL_DRAFT
    assert _pick_rich_notes(SEED_DRAFT, MANUAL_DRAFT) == MANUAL_DRAFT


def test_pick_rich_notes_allows_short_edit_and_clear():
    long_notes = MANUAL_DRAFT + (" more" * 40)
    assert _pick_rich_notes(long_notes, "<p>trimmed</p>") == "<p>trimmed</p>"
    assert _pick_rich_notes(long_notes, "") == ""


def test_pick_argument_notes_shorter_manual_wins():
    long_server = "<p>" + ("old outline " * 40) + "</p>"
    short_edit = "<p>short rewrite</p>"
    assert _pick_argument_notes(long_server, short_edit, "c3-s2-a") == short_edit


def test_protect_arguments_returns_rejection_reason():
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
                                        "notes": "<h2>Jackson's test</h2><p>manual</p>",
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
    out, reason = _protect_arguments_from_seed_or_thinner_push(_Cursor(existing), "ws", prepared)
    assert reason == "arguments_rejected_agent_jackson_title"
    notes = out["data"]["draftsBySide"]["petitioner"][0]["sections"][0]["prongs"][0]["notes"]
    assert "Jackson's test" in notes


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
                                    "title": "a. Under Youngstown, President falls into lowest Category",
                                    "notes": "<h2>Jackson's test</h2><p>manual outline notes</p>",
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
    assert "Jackson's test" in prong["notes"]


def test_merge_rejects_ai_jackson_title_and_claim_notes():
    server = {
        "draftsBySide": {
            "petitioner": [
                {
                    "id": "alt-q2-ladder",
                    "notes": MANUAL_DRAFT,
                    "sections": [
                        {
                            "id": "c3-s1",
                            "prongs": [
                                {
                                    "id": "c3-s1-a",
                                    "title": "a. Under Youngstown, President falls into lowest Category",
                                    "notes": "<h2>Jackson's test</h2><p>Category 1 out</p>",
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
                    "notes": MANUAL_DRAFT,
                    "sections": [
                        {
                            "id": "c3-s1",
                            "prongs": [
                                {
                                    "id": "c3-s1-a",
                                    "title": "a. Jackson’s method, not just his labels",
                                    "notes": (
                                        "<h2>The claim in one sentence</h2>"
                                        "<p>Where Congress has legislated in an area</p>"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }
    merged = _merge_arguments_boards(incoming, server)
    prong = merged["draftsBySide"]["petitioner"][0]["sections"][0]["prongs"][0]
    assert "Youngstown" in prong["title"]
    assert "Jackson's test" in prong["notes"]
    assert "claim in one sentence" not in prong["notes"].lower()


def test_stale_base_rejects_differing_push_without_matching_base():
    server_at = datetime(2026, 10, 6, 18, 46, 56, tzinfo=timezone.utc)
    existing = {
        "data": {"draftsBySide": {"petitioner": [{"id": "d1", "notes": "<p>mcp edit</p>"}]}},
        "updated_at": server_at,
    }
    prepared = {
        "kind": "arguments",
        "id": "main",
        "data": {"draftsBySide": {"petitioner": [{"id": "d1", "notes": "<p>stale tab</p>"}]}},
        "updatedAt": to_epoch_ms(server_at) + 2000,
        # Missing baseUpdatedAt: old client / dirty stamp race.
    }
    out, reason, server_row = _protect_arguments_stale_base(
        _Cursor(existing), "ws", prepared
    )
    assert reason == "arguments_rejected_missing_base"
    assert server_row["data"]["draftsBySide"]["petitioner"][0]["notes"] == "<p>mcp edit</p>"
    assert server_row["serverUpdatedAt"] == to_epoch_ms(server_at)
    assert out is prepared


def test_stale_base_allows_push_when_base_matches():
    server_at = datetime(2026, 10, 6, 18, 46, 56, tzinfo=timezone.utc)
    existing = {
        "data": {"draftsBySide": {"petitioner": [{"id": "d1", "notes": "<p>server</p>"}]}},
        "updated_at": server_at,
    }
    prepared = {
        "kind": "arguments",
        "id": "main",
        "data": {"draftsBySide": {"petitioner": [{"id": "d1", "notes": "<p>real edit</p>"}]}},
        "updatedAt": to_epoch_ms(server_at) + 5000,
        "baseUpdatedAt": to_epoch_ms(server_at),
    }
    out, reason, server_row = _protect_arguments_stale_base(
        _Cursor(existing), "ws", prepared
    )
    assert reason is None
    assert server_row is None
    assert out["data"]["draftsBySide"]["petitioner"][0]["notes"] == "<p>real edit</p>"


def test_stale_base_identical_content_skips_check():
    server_at = datetime(2026, 10, 6, 18, 46, 56, tzinfo=timezone.utc)
    board = {"draftsBySide": {"petitioner": [{"id": "d1", "notes": "<p>same</p>"}]}}
    existing = {"data": board, "updated_at": server_at}
    prepared = {
        "kind": "arguments",
        "id": "main",
        "data": board,
        "updatedAt": to_epoch_ms(server_at) + 9000,
    }
    out, reason, server_row = _protect_arguments_stale_base(
        _Cursor(existing), "ws", prepared
    )
    assert reason is None
    assert server_row is None
    assert out is prepared
