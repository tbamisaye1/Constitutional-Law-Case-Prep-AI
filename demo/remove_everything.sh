#!/usr/bin/env bash
# Remove ALL Erin/demo scaffolding so you can rebuild the RAG agent yourself.
# Run from repo root: ./demo/remove_everything.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_PY="$ROOT/app/main.py"
DEMO_DIR="$ROOT/demo"
export ROOT

echo "=== Case-Law-Agent demo removal ==="
echo "Repo: $ROOT"
echo

# 1. Clear indexed data
if [[ -f "$DEMO_DIR/teardown_demo.py" ]]; then
  echo "[1/4] Clearing FAISS index and cached JSON..."
  (cd "$ROOT" && python3 "$DEMO_DIR/teardown_demo.py") || true
else
  echo "[1/4] No teardown script; skipping data clear"
fi

# 2. Restore app/main.py (strip DEMO blocks + demo-only imports)
if [[ -f "$MAIN_PY" ]]; then
  echo "[2/4] Restoring app/main.py (removing /demo route)..."
  python3 - <<'PY'
import os
import re
from pathlib import Path

root = Path(os.environ["ROOT"])
main = root / "app" / "main.py"
text = main.read_text(encoding="utf-8")

# Remove marked demo blocks
text = re.sub(r"\n# >>> DEMO_START.*?# >>> DEMO_END\n*", "\n", text, flags=re.DOTALL)

# Drop imports only used by demo route
for line in (
    "from pathlib import Path\n",
    "from fastapi.responses import FileResponse\n",
):
    if line in text and "DEMO" not in text and "_DEMO_HTML" not in text:
        text = text.replace(line, "")

# Collapse extra blank lines after imports
text = re.sub(r"\n{3,}", "\n\n", text)
main.write_text(text, encoding="utf-8")
print(f"  Cleaned {main}")
PY
else
  echo "[2/4] main.py not found — skip"
fi

# 3. Remove demo gitignore line
GITIGNORE="$ROOT/.gitignore"
if [[ -f "$GITIGNORE" ]]; then
  echo "[3/4] Cleaning .gitignore..."
  grep -v 'demo/supreme_court_cases.json' "$GITIGNORE" > "${GITIGNORE}.tmp" || true
  mv "${GITIGNORE}.tmp" "$GITIGNORE"
fi

# 4. Delete demo folder contents (leave a stub)
echo "[4/4] Removing demo scripts and viewer..."
rm -f \
  "$DEMO_DIR/bootstrap_moot_index.py" \
  "$DEMO_DIR/chat_cli.py" \
  "$DEMO_DIR/teardown_demo.py" \
  "$DEMO_DIR/index.html" \
  "$DEMO_DIR/README.md" \
  "$DEMO_DIR/MANIFEST.md" \
  "$DEMO_DIR/supreme_court_cases.json" \
  "$DEMO_DIR/remove_everything.sh"

cat > "$DEMO_DIR/DEMO_REMOVED.txt" <<EOF
Demo scaffolding removed on $(date +%Y-%m-%d).

Rebuild the RAG agent yourself:
  - Case-Law-Agent/docs/ARCHITECTURE.md
  - Case-Law-Agent/docs/TODO.md
  - app/rag/ + app/agents/ + app/api/chat.py (still in repo)

Delete this file and the demo/ folder whenever you like.
EOF

echo
echo "Done. /demo route gone. FAISS index cleared."
echo "Core agent code (app/agents, app/rag, /chat) is still there for you to rebuild on."
echo "See demo/DEMO_REMOVED.txt for next steps."
