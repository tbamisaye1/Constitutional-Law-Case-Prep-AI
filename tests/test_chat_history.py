from app.grounding.prompts import (
    advocacy_side_instruction,
    build_reason_user_payload,
    format_chat_history,
)
from app.grounding.selection import build_chat_message


def test_format_chat_history_keeps_newest_within_budget():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "answer two"},
    ]
    text = format_chat_history(history, max_chars=80)
    assert "follow up" in text or "answer two" in text
    assert "User:" in text


def test_build_reason_payload_includes_history():
    payload = build_reason_user_payload(
        "what about standing?",
        "[ev-0] note\nstanding text",
        [{"role": "user", "content": "explain Padilla"}, {"role": "assistant", "content": "Padilla held…"}],
    )
    assert "CONVERSATION SO FAR" in payload
    assert "Padilla" in payload
    assert "what about standing?" in payload


def test_advocacy_side_instruction_locks_respondent():
    text = advocacy_side_instruction("respondent")
    assert "RESPONDENT" in text
    assert "petitioner" in text.lower()
    assert advocacy_side_instruction("both") == ""


def test_build_chat_message_includes_advocacy_side():
    packed = build_chat_message(
        "evaluate my arguments for respondent",
        side="respondent",
    )
    assert "ADVOCACY SIDE" in packed
    assert "RESPONDENT" in packed
    assert "evaluate my arguments for respondent" in packed


def test_build_reason_payload_includes_advocacy_side():
    payload = build_reason_user_payload(
        "fill the Youngstown holes",
        "[ev-0] note\nrespondent Youngstown notes",
        side="respondent",
    )
    assert "ADVOCACY SIDE" in payload
    assert "RESPONDENT" in payload
