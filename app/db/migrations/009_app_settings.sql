-- Key/value settings for server-owned defaults (MCP default workspace, etc.).
-- Values are JSONB so a setting can be a string, object, or list without a
-- schema change per key.

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE app_settings IS
    'Server-owned defaults. MCP uses mcp_default_workspace; other keys may '
    'follow. Not workspace-scoped.';
