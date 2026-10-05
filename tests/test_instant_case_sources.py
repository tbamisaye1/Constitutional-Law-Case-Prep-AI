from app.rag.store import is_instant_case_source


def test_is_instant_case_source():
    assert is_instant_case_source("Instant Case [Bronner v. USA] — ACFRog.pdf")
    assert is_instant_case_source(
        "ACFrOgAS9b_Udb_HXITuugoCYWQTS8KraNYH7aJFryKr1hOnjVFPza5bdOIpFa8rK68VfKHPjQhmpAmp3db7WWLnFKzvRRXf5WS4sczQLcDDN_Kbkf0qnJ6n.pdf"
    )
    assert is_instant_case_source("Bronner Joint Appendix.pdf")
    assert not is_instant_case_source("Hamdi v. Rumsfeld, 542 U.S. 507 (2004) (1).pdf")
    assert not is_instant_case_source("California v. Ciraolo (Oyez summary)")


def test_docs_for_instant_case_ranks_by_query_not_early_page(monkeypatch):
    """Regression: early-page bias returned only the caption for appellate asks."""
    from langchain_core.documents import Document

    from app.rag import store as store_mod

    caption = Document(
        page_content="IN THE UNITED STATES COURT OF APPEALS FOR THE FOURTEENTH CIRCUIT caption only",
        metadata={"source": "ACFrOgRecord.pdf", "page": 1},
    )
    reasoning = Document(
        page_content=(
            "We disagree. Because the president acted within his Article II authority "
            "under Youngstown Category One and the AUMF, the appellate court reversed."
        ),
        metadata={"source": "ACFrOgRecord.pdf", "page": 12},
    )

    class FakeStore:
        def similarity_search_with_score(self, query, k=10):
            # Hamdi-ish neighbor scores better on raw distance; Instant Case
            # reasoning must still win after the Instant Case filter.
            other = Document(
                page_content="Curtiss-Wright foreign affairs argument",
                metadata={"source": "United States v. Curtiss-Wright.pdf", "page": 5},
            )
            return [(other, 0.1), (caption, 0.4), (reasoning, 0.35)]

    hits = store_mod.docs_for_instant_case(
        FakeStore(),
        "Appellate Court reasoning president within authority Youngstown",
        limit=2,
    )
    assert hits
    assert "We disagree" in hits[0][0].page_content
    assert hits[0][0].metadata["page"] == 12
