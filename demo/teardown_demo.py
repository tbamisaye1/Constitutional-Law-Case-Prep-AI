"""
LOCAL DEMO ONLY — remove bootstrapped index and cached case JSON.

Does not touch app/ code. Safe to run when you are done showing the demo.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "data" / "faiss_index",
    ROOT / "demo" / "supreme_court_cases.json",
]


def main() -> None:
    for path in TARGETS:
        if path.is_dir():
            shutil.rmtree(path)
            print(f"Removed directory {path}")
        elif path.is_file():
            path.unlink()
            print(f"Removed file {path}")
        else:
            print(f"Skip (not found): {path}")
    print("Demo data cleared. Delete demo/ folder itself if you want zero trace.")


if __name__ == "__main__":
    main()
