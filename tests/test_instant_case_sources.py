from langchain_core.documents import Document

from app.rag.store import docs_for_instant_case, is_instant_case_source


def test_is_instant_case_source():
    assert is_instant_case_source("Instant Case [Bronner v. USA] — ACFRog.pdf")
    assert is_instant_case_source(
        "ACFrOgAS9b_Udb_HXITuugoCYWQTS8KraNYH7aJFryKr1hOnjVFPza5bdOIpFa8rK68VfKHPjQhmpAmp3db7WWLnFKzvRRXf5WS4sczQLcDDN_Kbkf0qnJ6n.pdf"
    )
    assert is_instant_case_source("Bronner Joint Appendix.pdf")
    assert not is_instant_case_source("Hamdi v. Rumsfeld, 542 U.S. 507 (2004) (1).pdf")
    assert not is_instant_case_source("California v. Ciraolo (Oyez summary)")


def test_docs_for_instant_case_prefers_doctrine_pages_over_caption():
    """Caption pages must not crowd out Article II / Youngstown reasoning."""
    from app.rag import store as store_mod

    caption = Document(
        page_content="IN THE UNITED STATES COURT OF APPEALS FOR THE FOURTEENTH CIRCUIT Bobby Bronner caption only",
        metadata={"source": "Instant Case [Bronner v. USA] — ACFrOg.pdf", "page": 1},
    )
    reasoning = Document(
        page_content=(
            "IV Article II Analysis. We disagree. Because the president acted within his "
            "Article II authority under Youngstown Category One and the AUMF and NDAA, "
            "the government arguments on detention authority are affirmed."
        ),
        metadata={"source": "Instant Case [Bronner v. USA] — ACFrOg.pdf", "page": 12},
    )

    class FakeStore:
        def similarity_search_with_score(self, query, k=10):
            other = Document(
                page_content="Curtiss-Wright foreign affairs argument",
                metadata={"source": "United States v. Curtiss-Wright.pdf", "page": 5},
            )
            # Caption looks closer in embedding space; lexical path must still win.
            return [(other, 0.1), (caption, 0.2), (reasoning, 0.55)]

    # Patch document scan used by lexical ranking.
    original = store_mod._all_documents
    store_mod._all_documents = lambda _store: [caption, reasoning]
    try:
        hits = docs_for_instant_case(
            FakeStore(),
            "Appellate Court reasoning president within authority Youngstown Article II",
            limit=2,
        )
    finally:
        store_mod._all_documents = original

    assert hits
    assert hits[0][0].metadata["page"] == 12
    assert "Article II" in hits[0][0].page_content
