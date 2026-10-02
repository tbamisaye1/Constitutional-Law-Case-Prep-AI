from app.grounding.prompts import build_reason_user_payload, format_chat_history


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
