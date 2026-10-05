from app.rag.store import is_instant_case_source


def test_is_instant_case_source():
    assert is_instant_case_source("Instant Case [Bronner v. USA] — ACFRog.pdf")
    assert is_instant_case_source("ACFrOgAS9b_Udb_HXITuugoCYWQTS8KraNYH7aJFryKr1hOnjVFPza5bdOIpFa8rK68VfKHPjQhmpAmp3db7WWLnFKzvRRXf5WS4sczQLcDDN_Kbkf0qnJ6n.pdf")
    assert is_instant_case_source("Bronner Joint Appendix.pdf")
    assert not is_instant_case_source("Hamdi v. Rumsfeld, 542 U.S. 507 (2004) (1).pdf")
    assert not is_instant_case_source("California v. Ciraolo (Oyez summary)")
