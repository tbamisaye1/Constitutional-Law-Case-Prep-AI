-- Durable full-workspace snapshots for competition-safe recovery.
--
-- library_record_revisions keeps per-row history on overwrite. This table
-- stores named whole-workspace JSON dumps (notes, arguments, annotations,
-- cases, document metadata) that survive hard refresh, sync wipes, and
-- mistaken deletes. Auto backups are created from the sync path; manual
-- backups are created from the UI Download / Backup buttons.

CREATE TABLE IF NOT EXISTS workspace_backups (
    id BIGSERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS workspace_backups_lookup_idx
    ON workspace_backups (workspace_id, created_at DESC);

COMMENT ON TABLE workspace_backups IS
    'Full workspace JSON snapshots for recovery. Not part of the sync loop.';
