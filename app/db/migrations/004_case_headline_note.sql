-- Short one-liner on the case card (above holding).
ALTER TABLE cases
    ADD COLUMN IF NOT EXISTS headline_note TEXT NOT NULL DEFAULT '';
