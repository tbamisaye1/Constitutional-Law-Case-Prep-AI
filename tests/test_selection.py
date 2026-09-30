from app.grounding.selection import (
    build_chat_message,
    extract_selection,
    extract_user_question,
    retrieval_query,
)


def test_build_chat_message_includes_selection():
    packed = build_chat_message(
        "what does this mean",
        "On the same day, the Court in Rumsfeld v. Padilla overturned…",
    )
    assert "Rumsfeld v. Padilla" in packed
    assert "what does this mean" in packed
    assert "SELECTED PASSAGE" in packed


def test_retrieval_query_prefers_selected_passage():
    packed = build_chat_message(
        "what does this mean",
        "On the same day, the Court in Rumsfeld v. Padilla overturned a lower court’s grant of habeas.",
    )
    query = retrieval_query(packed)
    assert "Padilla" in query
    assert "what does this mean" in query
    # Passage should lead so embedding weights case names first.
    assert query.index("Padilla") < query.index("what does this mean")


def test_extract_helpers_round_trip():
    packed = build_chat_message("Define AUMF", "Authorization for Use of Military Force")
    assert extract_selection(packed) == "Authorization for Use of Military Force"
    assert extract_user_question(packed) == "Define AUMF"
    assert extract_selection("plain question only") == ""
    assert extract_user_question("plain question only") == "plain question only"


def test_build_without_selection_is_plain():
    assert build_chat_message("hello", "") == "hello"
    assert build_chat_message("hello", None) == "hello"
