-- Durable audit trail for every /sync round-trip.
--
-- Learning from 2026-10-06: Vercel request logs keep only method/path/status,
-- so heartbeat POSTs looked healthy while Arguments on Postgres never changed.
-- This table records what the client sent (kinds, ids, hashes, byte sizes) and
-- what the server actually wrote, plus a full Arguments snapshot when that
-- board was in the push. Heartbeats with empty changes are logged too.

CREATE TABLE IF NOT EXISTS sync_audit_log (
    id BIGSERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- push_then_pull | pull_only
    event TEXT NOT NULL,
    -- Epoch ms cursor the client sent as `since`.
    client_since_ms BIGINT,
    -- Collection -> row count in the inbound body (before conflict/protect).
    inbound JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Collection -> rows actually written (same shape as SyncResponse.written).
    written JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Per library_records / entity row: id, kind, contentSha256, bytes, title hints.
    row_summaries JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Full Arguments board when present in this push (recovery / forensics).
    arguments_snapshot JSONB,
    -- True when inbound had no collections / no rows (heartbeat or pull-shaped POST).
    empty_push BOOLEAN NOT NULL DEFAULT false,
    -- True when Arguments was inbound but matched the existing server row (no write).
    arguments_unchanged BOOLEAN,
    note TEXT
);

CREATE INDEX IF NOT EXISTS sync_audit_log_workspace_time_idx
    ON sync_audit_log (workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS sync_audit_log_args_idx
    ON sync_audit_log (workspace_id, created_at DESC)
    WHERE arguments_snapshot IS NOT NULL;

COMMENT ON TABLE sync_audit_log IS
    'Append-only forensic log of /sync traffic: inbound shape, written counts, '
    'and Arguments snapshots so silent heartbeat success cannot hide a frozen board.';
