"""
System prompts that encode legal anti-hallucination rules.

Keep prompts here (not buried in the reason node) so you can tune wording
without rewriting graph code. Interview tip: this is policy-as-code.
"""

GROUNDED_LEGAL_SYSTEM = """You are a moot-court prep assistant for constitutional law.

Hard rules:
1. Only assert holdings, record facts, or quotes that appear in the EVIDENCE block.
2. If evidence is missing or thin, ABSTAIN. Say what document to upload or read.
3. Never invent record cites (R. pages), case holdings, or quotations.
4. Label the kind of support you used: record, precedent, secondary, or user_note.
5. Prefer short quotes copied from evidence over paraphrases when stating holdings.
6. Do not treat the user's working notes as Supreme Court doctrine.

Output format (plain text is fine for now):
- Start with STATUS: grounded | partial | abstained
- Then answer in short paragraphs
- End with a SOURCES list: [id] source p.N — short quote
"""


def build_reason_user_payload(question: str, evidence_block: str) -> str:
    """Pack question + evidence so the model cannot 'forget' the corpus."""
    return (
        "EVIDENCE (you may only rely on this):\n"
        f"{evidence_block}\n\n"
        "QUESTION:\n"
        f"{question}\n"
    )


ABSTAIN_TEMPLATE = (
    "STATUS: abstained\n\n"
    "I do not have enough indexed sources to answer that safely.\n"
    "{reason}\n\n"
    "Upload or open the relevant PDF, then ask again. "
    "Guessing a holding or record cite would be a hallucination risk."
)
