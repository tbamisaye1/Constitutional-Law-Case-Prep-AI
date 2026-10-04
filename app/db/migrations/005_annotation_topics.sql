-- Per-article topic tags on annotations (multi-select labels, JSON array).
ALTER TABLE annotations
    ADD COLUMN IF NOT EXISTS topics JSONB NOT NULL DEFAULT '[]'::jsonb;
