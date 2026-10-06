-- Keep prior library_records payloads so a bad sync or seed wipe can be undone.
--
-- Sync still last-write-wins on library_records itself. This table is append-only
-- history: every successful overwrite of a library row archives the previous
-- JSON first. Clients do not pull these rows in the normal sync loop.

CREATE TABLE IF NOT EXISTS library_record_revisions (
    id BIGSERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    record_id TEXT NOT NULL,
    data JSONB NOT NULL,
    row_updated_at TIMESTAMPTZ,
    revised_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source TEXT NOT NULL DEFAULT 'sync_push'
);

CREATE INDEX IF NOT EXISTS library_record_revisions_lookup_idx
    ON library_record_revisions (workspace_id, kind, record_id, revised_at DESC);

COMMENT ON TABLE library_record_revisions IS
    'Append-only archive of library_records.data before each overwrite. '
    'Used for recovery (arguments, notebook, guide edits, etc.).';
