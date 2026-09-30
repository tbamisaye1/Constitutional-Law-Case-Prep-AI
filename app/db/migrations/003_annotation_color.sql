-- Optional highlight color for PDF annotations (gold, green, blue, rose, violet).
ALTER TABLE annotations
    ADD COLUMN IF NOT EXISTS color TEXT NOT NULL DEFAULT 'gold';
