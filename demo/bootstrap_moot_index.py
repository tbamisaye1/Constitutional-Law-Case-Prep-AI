"""
LOCAL DEMO ONLY — bootstrap FAISS from Oyez-style case summaries.

Indexes the moot-court precedents listed in ../../reference/moot-court-cases-readme.txt
using supreme_court_cases.json (CaseFacts GitHub). Run once before demo_chat_cli.py.

To remove later: delete demo/ and data/faiss_index/, or run demo/teardown_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Project root = Constitutional-Law-Case-Prep-AI/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.chunking import chunk_text
from app.rag.store import build_or_merge_store

CASES_URL = (
    "https://raw.githubusercontent.com/idirlab/CaseFacts/main/"
    "dataset/supreme_court_cases.json"
)

# Names as they appear in supreme_court_cases.json (verified 2026-08-24)
MOOT_CASE_NAMES = [
    "Youngstown Sheet & Tube Company v. Sawyer",
    "Ex parte Milligan",
    "Mathews v. Eldridge",
    "United States v. Jones",
    "California v. Ciraolo",
]

# Fallback: also try legal-citation-support copy if present
LOCAL_JSON_CANDIDATES = [
    ROOT / "demo" / "supreme_court_cases.json",
    Path(__file__).resolve().parents[3]
    / "AI-projects"
    / "legal-citation-support"
    / "data"
    / "raw"
    / "supreme_court_cases.json",
]


def ensure_cases_json() -> Path:
    dest = ROOT / "demo" / "supreme_court_cases.json"
    for candidate in LOCAL_JSON_CANDIDATES:
        if candidate.exists():
            if candidate != dest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(candidate.read_bytes())
            return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading case summaries to {dest}")
    subprocess.run(["curl", "-fL", CASES_URL, "-o", str(dest)], check=True)
    return dest


def case_passage(case: dict) -> str:
    title = case.get("name") or "Unknown"
    facts = (case.get("facts") or "").strip()
    conclusion = (case.get("api_conclusion") or "").strip()
    parts = [f"Case: {title}"]
    if facts:
        parts.append(f"Facts: {facts}")
    if conclusion:
        parts.append(f"Holding summary: {conclusion}")
    return "\n\n".join(parts)


def main() -> None:
    cases_path = ensure_cases_json()
    all_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    by_name = {c.get("name", ""): c for c in all_cases}

    chunks = []
    found = []
    missing = []
    for name in MOOT_CASE_NAMES:
        case = by_name.get(name)
        if not case:
            missing.append(name)
            continue
        text = case_passage(case)
        chunks.extend(chunk_text(text, source=f"{name} (Oyez summary)", page=1))
        found.append(name)

    if not chunks:
        print("No cases indexed. Missing:", missing)
        sys.exit(1)

    build_or_merge_store(chunks)
    print(f"Indexed {len(found)} cases into {ROOT / 'data' / 'faiss_index'}")
    for n in found:
        print(f"  + {n}")
    if missing:
        print("Not in JSON (upload PDFs later):", missing)


if __name__ == "__main__":
    main()
