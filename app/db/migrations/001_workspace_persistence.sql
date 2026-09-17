-- Workspace-scoped persistence for moot prep data.
--
-- Shape notes that are easy to get wrong later:
--
-- 1. Row ids are client-generated TEXT, not server sequences. The frontend
--    already mints ids like 'pdf-1737...' and 'a-1737...' offline, before it
--    can reach the API, so the client has to own the id space. Every table is
--    therefore keyed on (workspace_id, id).
--
-- 2. Deletes are soft. A hard DELETE is invisible to a client that has been
--    offline, and it would silently resurrect the row on the next push. Setting
--    deleted_at lets the sync pull carry the tombstone.
--
-- 3. updated_at drives sync. Clients pull everything newer than their cursor
--    and pushes win by comparing updated_at, so each table needs an index on
--    (workspace_id, updated_at).

CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE workspaces IS
    'One row per browser-generated workspace key. There is no login yet; the '
    'client sends its key in the X-Workspace-Id header. When real accounts '
    'arrive, add an owner column and claim existing rows by key.';

CREATE TABLE IF NOT EXISTS matters (
    workspace_id UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    season TEXT NOT NULL DEFAULT '',
    -- Free-form list of {id, label, short}. Issue structure still changes with
    -- each moot problem, so it is not worth its own table yet.
    issues JSONB NOT NULL DEFAULT '[]'::JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (workspace_id, id)
);

CREATE INDEX IF NOT EXISTS matters_workspace_updated_idx
    ON matters (workspace_id, updated_at);

CREATE TABLE IF NOT EXISTS cases (
    workspace_id UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    cite TEXT NOT NULL DEFAULT '',
    year TEXT NOT NULL DEFAULT '',
    -- Which question presented this case speaks to (1 or 2 today).
    issue INTEGER,
    tag TEXT,
    usefulness TEXT NOT NULL DEFAULT 'background',
    holding TEXT NOT NULL DEFAULT '',
    rule TEXT NOT NULL DEFAULT '',
    use_petitioner TEXT NOT NULL DEFAULT '',
    use_respondent TEXT NOT NULL DEFAULT '',
    suggested_file TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (workspace_id, id)
);

CREATE INDEX IF NOT EXISTS cases_workspace_updated_idx
    ON cases (workspace_id, updated_at);

CREATE TABLE IF NOT EXISTS documents (
    workspace_id UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    -- Plain text rather than a foreign key: 'case-at-bar' is a reserved id that
    -- has no row in cases, and a PDF can be attached before its case is saved.
    case_id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    size_bytes BIGINT NOT NULL DEFAULT 0,
    content_type TEXT NOT NULL DEFAULT 'application/pdf',
    -- Where the bytes live in Vercel Blob. Null means metadata synced from a
    -- client that had not finished uploading yet, so the reader should fall
    -- back to its local IndexedDB copy.
    --
    -- The pathname carries a random segment, so this URL is unguessable but it
    -- is not access-controlled: anyone holding it can read the PDF. That
    -- matches the rest of the app, which has no accounts yet. Moving to private
    -- Blob access is part of the auth work in docs/TODO.md.
    blob_pathname TEXT,
    blob_url TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (workspace_id, id)
);

CREATE INDEX IF NOT EXISTS documents_workspace_updated_idx
    ON documents (workspace_id, updated_at);

CREATE TABLE IF NOT EXISTS annotations (
    workspace_id UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    -- Which PDF the annotation sits on. Older annotations predate per-file
    -- tracking and carry null, which the reader treats as "the active file".
    document_id TEXT,
    page INTEGER NOT NULL DEFAULT 1,
    -- 'highlight' (has quote and rects) or 'page' (a free note on one page).
    kind TEXT NOT NULL DEFAULT 'page',
    quote TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    -- Normalised selection rectangles from react-pdf, used to redraw the
    -- highlight. Their shape is owned by the viewer, so keep them opaque here.
    rects JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (workspace_id, id)
);

CREATE INDEX IF NOT EXISTS annotations_workspace_updated_idx
    ON annotations (workspace_id, updated_at);

CREATE INDEX IF NOT EXISTS annotations_workspace_case_page_idx
    ON annotations (workspace_id, case_id, page);

CREATE TABLE IF NOT EXISTS notes (
    workspace_id UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    case_id TEXT NOT NULL,
    -- Note layer within a case: 'overview', 'holding', and friends from
    -- emptyLayerNotes() on the client.
    layer_id TEXT NOT NULL,
    -- TipTap output. Stored as HTML because that is what the editor round-trips
    -- losslessly; do not try to parse it server-side.
    html TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (workspace_id, case_id, layer_id)
);

CREATE INDEX IF NOT EXISTS notes_workspace_updated_idx
    ON notes (workspace_id, updated_at);

CREATE TABLE IF NOT EXISTS library_records (
    workspace_id UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    -- 'opinions' | 'case_facts' | 'cites' | 'timeline'
    kind TEXT NOT NULL,
    id TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (workspace_id, kind, id)
);

COMMENT ON TABLE library_records IS
    'Research rows the backend only stores and hands back: opinions, case '
    'facts, citations, timeline events. They are kept as JSONB because their '
    'shape is still driven by the seed files on the client and the agent does '
    'not query them yet. Promote a kind to its own table once something '
    'server-side needs to filter or join on its fields.';

CREATE INDEX IF NOT EXISTS library_records_workspace_updated_idx
    ON library_records (workspace_id, updated_at);
