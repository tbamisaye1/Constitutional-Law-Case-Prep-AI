# Constitutional Law Case Prep AI (backend)

**Author:** Tobi Bamisaye

Python FastAPI backend for moot prep: PDF ingest, FAISS RAG, LangGraph prep agent, Word export. Models via OpenRouter (OpenAI-compatible API).

Frontend: [Constitutional-Law-Case-Prep](https://github.com/tbamisaye1/Constitutional-Law-Case-Prep).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# put OPENROUTER_API_KEY in .env
uvicorn app.main:app --reload --port 8000
```

Health check: http://127.0.0.1:8000/health

The health response reports whether Postgres is reachable and whether any
migrations are pending, so it is the first thing to read when sync looks off.

## Persistence (Neon Postgres + Vercel Blob)

Notes, annotations, case cards, matters, and uploaded PDFs persist per
workspace. Design and trade-offs: `docs/ADR-001-backend-persistence.md`.

### First-time setup

```bash
# 1. Provision Neon on this Vercel project (opens a browser)
vercel integration add neon

# 2. Pull the credentials it wrote, plus the Blob token
vercel env pull .env.local

# 3. Create the tables
python -m app.db.migrate
```

Use the **pooled** connection string as `DATABASE_URL`, the one whose host
contains `-pooler`. The unpooled endpoint runs out of connections under
serverless traffic.

Settings read `.env` and `.env.local`, with `.env.local` winning, so the
credentials `vercel env pull` writes are picked up without copying them across.

### Migrations

```bash
python -m app.db.migrate            # apply anything pending
python -m app.db.migrate --status   # list applied and pending, change nothing
```

Forward-only and idempotent, so re-running is safe. They are deliberately not
applied on startup: every cold start would race the same DDL, and a broken
migration would take the API down rather than one deploy step. Run this after
any deploy that adds a migration.

### Workspace identity

There are no accounts yet. The browser generates a UUID, keeps it in
localStorage, and sends it as `X-Workspace-Id`; every row is keyed on it. This
separates one person's prep from another's, and copying the key to a second
browser opens the same notes there. It is not a password: anyone holding a key
can read that workspace.

### Endpoints

| Route | Purpose |
|-------|---------|
| `POST /sync` | Push changed rows and pull everything since the client's cursor |
| `GET /sync` | Pull only, for a device that has never synced |
| `GET /sync/status` | Live row counts for the workspace |
| `POST /documents` | Store a PDF's bytes in Blob (4 MB cap, see below) |
| `GET /documents/{id}/file` | 307 redirect to the PDF in Blob |
| `DELETE /documents/{id}` | Tombstone the document and delete its bytes |

Uploads are capped at 4 MB because Vercel rejects request bodies over 4.5 MB
before our code runs. Downloads redirect rather than stream for the same reason
in reverse: a response body is capped at 4.5 MB too. Lifting the upload cap
means client-direct uploads to Blob, which is still open (`docs/TODO.md`).

## Tests

```bash
pip install -e '.[dev]'

# Unit tests that need no database
pytest

# Plus the SQL-level tests: conflict resolution, tombstones, workspace isolation
createdb case_law_test
export TEST_DATABASE_URL='postgresql://localhost/case_law_test?sslmode=disable'
pytest
```

The persistence tests run against a real Postgres rather than a mocked cursor,
because the behaviour worth testing lives in `ON CONFLICT` clauses that a mock
would accept while being wrong. They skip when `TEST_DATABASE_URL` is unset.

## Layout

```
app/
  main.py           FastAPI app + CORS
  config.py         env settings
  api/              route modules
  agents/           LangGraph state + prep graph
  rag/              chunk + FAISS store
  grounding/        claim verify + abstain
  llm/              OpenRouter client
  export/           notes → docx
  db/               Neon connection, SQL migrations, sync queries
  storage/          Vercel Blob client (PDF bytes + FAISS index zip)
```
