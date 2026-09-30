-- Pin flag for annotations so a note can stay at the top of the panel.
ALTER TABLE annotations
    ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT false;
