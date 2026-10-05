from app.grounding.selection import (
    build_chat_message,
    expand_instant_case_aliases,
    extract_selection,
    extract_source_file,
    extract_source_page,
    extract_user_question,
    mentions_instant_case,
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


def test_build_includes_source_article_and_page():
    packed = build_chat_message(
        'what does "on the basis of standing" mean',
        "but the decision has been reversed on appeal on the basis of standing",
        source_file="Detention of U.S. Persons as Enemy Belligerents.pdf",
        page=2,
    )
    assert "SOURCE ARTICLE: Detention of U.S. Persons as Enemy Belligerents.pdf (p. 2)" in packed
    assert extract_source_file(packed).endswith(".pdf")
    assert extract_source_page(packed) == 2
    assert "reading to understand" in packed.lower() or "not arguing petitioner" in packed.lower()


def test_retrieval_query_prefers_selected_passage():
    packed = build_chat_message(
        "what does this mean",
        "On the same day, the Court in Rumsfeld v. Padilla overturned a lower court’s grant of habeas.",
        source_file="padilla-note.pdf",
        page=3,
    )
    query = retrieval_query(packed)
    assert "Padilla" in query
    assert "what does this mean" in query
    assert "padilla-note.pdf" in query
    # Passage should lead so embedding weights case names first.
    assert query.index("Padilla") < query.index("what does this mean")


def test_extract_helpers_round_trip():
    packed = build_chat_message("Define AUMF", "Authorization for Use of Military Force")
    assert extract_selection(packed) == "Authorization for Use of Military Force"
    assert extract_user_question(packed) == "Define AUMF"
    assert extract_selection("plain question only") == ""
    assert extract_user_question("plain question only") == "plain question only"
    assert extract_source_file("plain question only") == ""
    assert extract_source_page("plain question only") is None


def test_build_without_selection_is_plain():
    assert build_chat_message("hello", "") == "hello"
    assert build_chat_message("hello", None) == "hello"
    assert build_chat_message("hello", None, source_file="x.pdf", page=1) == "hello"
    packed = build_chat_message("facts in the instant case", "")
    assert "Bronner" in packed
    assert "instant case" in packed.lower()


def test_instant_case_aliases_mean_bronner():
    assert mentions_instant_case("facts in the instant case")
    assert mentions_instant_case("what does the case at bar say about standing")
    assert mentions_instant_case("Go throguht the instatnt case, what were gov arguments")
    assert mentions_instant_case("bronner appellate reasoning")
    assert not mentions_instant_case("summarize Hamdi")
    expanded = expand_instant_case_aliases("standing in the instant case")
    assert "Bronner" in expanded
    assert "Joint Appendix" in expanded
    query = retrieval_query("standing issues in the instant case")
    assert "Bronner" in query
    assert "instant case" in query.lower()
    # Typo path must expand too, or Instant Case boost never runs.
    typo_q = retrieval_query(
        "Go throguht the instatnt case, Appellate Court reasoning for president authority"
    )
    assert "Bronner" in typo_q

