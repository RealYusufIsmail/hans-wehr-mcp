-- Hans Wehr MCP — SQLite Schema
-- All Arabic text stored UTF-8. Two forms per Arabic field:
--   *          → with full tashkeel/diacritics as extracted from PDF
--   *_unvoweled → stripped of U+064B–U+065F and U+0670 for flexible search
--
-- Enable WAL mode and foreign keys at connection time:
--   PRAGMA journal_mode=WAL;
--   PRAGMA foreign_keys=ON;

-- ---------------------------------------------------------------------------
-- roots
-- One row per Arabic root (tri-literal or quad-literal headword).
-- These are the bold, large-font entries that organise the dictionary.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS roots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Arabic root in script, with tashkeel if present in the PDF
    arabic              TEXT    NOT NULL,
    -- Same root stripped of diacritics — used for lookup normalisation
    arabic_unvoweled    TEXT    NOT NULL,
    -- Academic transliteration per Hans Wehr convention (ḥ, ḫ, ṭ, ẓ, ā, ī, ū, …)
    transliteration     TEXT    NOT NULL,
    -- ASCII-safe transliteration: Latin combining diacritics stripped (ā→a, ḥ→h, ṭ→t …)
    -- Populated by import_db.py. Lets users type without special characters.
    transliteration_ascii TEXT  NOT NULL DEFAULT '',

    -- PDF page number where this root heading first appears
    page_number         INTEGER NOT NULL,

    -- Denormalised entry count — updated by import_db.py after all entries are loaded
    entry_count         INTEGER NOT NULL DEFAULT 0,

    -- Ensure we don't duplicate roots (normalise against unvoweled form)
    UNIQUE (arabic_unvoweled)
);

CREATE INDEX IF NOT EXISTS idx_roots_arabic_unvoweled    ON roots (arabic_unvoweled);
CREATE INDEX IF NOT EXISTS idx_roots_transliteration     ON roots (transliteration);
CREATE INDEX IF NOT EXISTS idx_roots_transliteration_asc ON roots (transliteration_ascii);
CREATE INDEX IF NOT EXISTS idx_roots_page                ON roots (page_number);

-- ---------------------------------------------------------------------------
-- entries
-- One row per lexical item (derived form, noun, adjective, phrase, etc.)
-- under a root. This is the core table.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Foreign key to the parent root
    root_id             INTEGER NOT NULL REFERENCES roots (id) ON DELETE CASCADE,

    -- Entry word(s) in Arabic script (may be a compound or phrase)
    arabic              TEXT    NOT NULL,
    arabic_unvoweled    TEXT    NOT NULL,

    -- Academic transliteration
    transliteration     TEXT,
    -- ASCII-safe transliteration for casual search (ā→a, ḥ→h, ṭ→t …)
    transliteration_ascii TEXT,

    -- Grammatical part of speech
    -- Allowed values: verb | noun | adjective | adverb | particle | phrase | proper_noun
    part_of_speech      TEXT,

    -- For verbs: Roman numeral form I–X (stored as text "I", "II", …, "X")
    verb_form           TEXT,

    -- JSON array of plural forms e.g. '["كُتُب","أَكْتَاب"]'
    -- NULL if not applicable (verbs, adjectives without broken plurals listed)
    plural_forms        TEXT,

    -- Full English definition text as printed in the dictionary
    definition          TEXT    NOT NULL,

    -- Extra grammatical notes printed inline, e.g. "with acc.", "foll. by bi-"
    grammar_notes       TEXT,

    -- Source PDF page — essential for audit trail
    page_number         INTEGER NOT NULL,

    -- Confidence score assigned by the structural parser (0.0–1.0)
    -- 1.0 = perfectly clean parse; <0.75 = flagged for review
    confidence          REAL    NOT NULL DEFAULT 1.0,

    -- 1 if this entry needs human or LLM review; 0 otherwise
    needs_review        INTEGER NOT NULL DEFAULT 0 CHECK (needs_review IN (0, 1)),

    -- Insertion timestamp (ISO 8601)
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_entries_root_id              ON entries (root_id);
CREATE INDEX IF NOT EXISTS idx_entries_arabic_unvoweled     ON entries (arabic_unvoweled);
CREATE INDEX IF NOT EXISTS idx_entries_transliteration_asc  ON entries (transliteration_ascii);
CREATE INDEX IF NOT EXISTS idx_entries_page                 ON entries (page_number);
CREATE INDEX IF NOT EXISTS idx_entries_needs_review         ON entries (needs_review) WHERE needs_review = 1;
CREATE INDEX IF NOT EXISTS idx_entries_confidence           ON entries (confidence);

-- ---------------------------------------------------------------------------
-- cross_references
-- Directed links between entries: "see →", "cf.", "plural of", etc.
-- Populated in two passes:
--   1. Import pass stores to_arabic_raw from the raw text.
--   2. Resolution pass matches to_arabic_raw → entries.id and sets to_entry_id.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cross_references (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The entry that contains the reference
    from_entry_id   INTEGER NOT NULL REFERENCES entries (id) ON DELETE CASCADE,

    -- The resolved target entry — NULL until the resolution pass completes
    to_entry_id     INTEGER REFERENCES entries (id) ON DELETE SET NULL,

    -- Raw Arabic target text extracted from the PDF (before resolution)
    to_arabic_raw   TEXT    NOT NULL,

    -- Reference type
    -- Allowed values: see | cf | plural_of | root_of | variant_of
    ref_type        TEXT    NOT NULL DEFAULT 'see',

    -- 1 once to_entry_id has been successfully filled
    resolved        INTEGER NOT NULL DEFAULT 0 CHECK (resolved IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_xref_from_entry ON cross_references (from_entry_id);
CREATE INDEX IF NOT EXISTS idx_xref_to_entry   ON cross_references (to_entry_id);
CREATE INDEX IF NOT EXISTS idx_xref_unresolved ON cross_references (resolved) WHERE resolved = 0;

-- ---------------------------------------------------------------------------
-- parse_metadata
-- One row per entry. Stores the raw extracted text and provenance so that
-- any parsed result can always be traced back to its source bytes.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parse_metadata (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Entry this metadata belongs to
    entry_id            INTEGER NOT NULL UNIQUE REFERENCES entries (id) ON DELETE CASCADE,

    -- Verbatim text extracted from the PDF before any parsing
    raw_text            TEXT    NOT NULL,

    -- How the entry was parsed
    -- structural   = font-metadata structural parser only
    -- llm_refined  = low-confidence entry corrected by LLM
    -- manual       = human-edited via review CLI
    parse_method        TEXT    NOT NULL DEFAULT 'structural'
                        CHECK (parse_method IN ('structural', 'llm_refined', 'manual')),

    -- Model used during LLM refinement (NULL for structural or manual)
    llm_model           TEXT,

    -- Mirrors entries.confidence for quick reporting without joining
    confidence          REAL    NOT NULL DEFAULT 1.0,

    -- JSON array of warning strings emitted during parsing
    -- e.g. ["no_transliteration", "page_break_split", "unknown_pos_tag"]
    extraction_warnings TEXT    NOT NULL DEFAULT '[]',

    -- ISO 8601 timestamp of when this row was created/updated
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_parse_meta_entry_id     ON parse_metadata (entry_id);
CREATE INDEX IF NOT EXISTS idx_parse_meta_parse_method ON parse_metadata (parse_method);

-- Keep updated_at current on any modification
CREATE TRIGGER IF NOT EXISTS trg_parse_metadata_updated_at
    AFTER UPDATE ON parse_metadata
    FOR EACH ROW
BEGIN
    UPDATE parse_metadata SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- ---------------------------------------------------------------------------
-- FTS5 virtual tables
-- Separate content-less FTS tables so the main tables stay normalised.
-- Populated by scripts/import_db.py after all entries are inserted.
-- ---------------------------------------------------------------------------

-- Arabic full-text search (unvoweled so queries don't need diacritics)
CREATE VIRTUAL TABLE IF NOT EXISTS entries_arabic_fts USING fts5(
    arabic_unvoweled,   -- searchable Arabic text (no diacritics)
    transliteration,    -- also searchable via transliteration
    entry_id UNINDEXED, -- link back to entries.id
    content='entries',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

-- English definition full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS entries_english_fts USING fts5(
    definition,         -- full English definition
    grammar_notes,      -- grammatical notes are also useful in search
    entry_id UNINDEXED,
    content='entries',
    content_rowid='id',
    tokenize='porter unicode61'  -- Porter stemming for English
);

-- Root full-text search (transliteration + arabic_unvoweled)
-- Lets lookup_root("kataba") and list_roots("k") match via FTS even with diacritics.
CREATE VIRTUAL TABLE IF NOT EXISTS roots_fts USING fts5(
    arabic_unvoweled,
    transliteration,
    transliteration_ascii,
    root_id UNINDEXED,
    content='roots',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS trg_roots_fts_insert
    AFTER INSERT ON roots
BEGIN
    INSERT INTO roots_fts (rowid, arabic_unvoweled, transliteration, transliteration_ascii, root_id)
    VALUES (NEW.id, NEW.arabic_unvoweled, NEW.transliteration, NEW.transliteration_ascii, NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS trg_roots_fts_delete
    AFTER DELETE ON roots
BEGIN
    INSERT INTO roots_fts (roots_fts, rowid, arabic_unvoweled, transliteration, transliteration_ascii, root_id)
    VALUES ('delete', OLD.id, OLD.arabic_unvoweled, OLD.transliteration, OLD.transliteration_ascii, OLD.id);
END;

CREATE TRIGGER IF NOT EXISTS trg_roots_fts_update
    AFTER UPDATE ON roots
BEGIN
    INSERT INTO roots_fts (roots_fts, rowid, arabic_unvoweled, transliteration, transliteration_ascii, root_id)
    VALUES ('delete', OLD.id, OLD.arabic_unvoweled, OLD.transliteration, OLD.transliteration_ascii, OLD.id);
    INSERT INTO roots_fts (rowid, arabic_unvoweled, transliteration, transliteration_ascii, root_id)
    VALUES (NEW.id, NEW.arabic_unvoweled, NEW.transliteration, NEW.transliteration_ascii, NEW.id);
END;

-- Keep FTS tables in sync with entries table
CREATE TRIGGER IF NOT EXISTS trg_entries_arabic_fts_insert
    AFTER INSERT ON entries
BEGIN
    INSERT INTO entries_arabic_fts (rowid, arabic_unvoweled, transliteration, entry_id)
    VALUES (NEW.id, NEW.arabic_unvoweled, NEW.transliteration, NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS trg_entries_arabic_fts_delete
    AFTER DELETE ON entries
BEGIN
    INSERT INTO entries_arabic_fts (entries_arabic_fts, rowid, arabic_unvoweled, transliteration, entry_id)
    VALUES ('delete', OLD.id, OLD.arabic_unvoweled, OLD.transliteration, OLD.id);
END;

CREATE TRIGGER IF NOT EXISTS trg_entries_arabic_fts_update
    AFTER UPDATE ON entries
BEGIN
    INSERT INTO entries_arabic_fts (entries_arabic_fts, rowid, arabic_unvoweled, transliteration, entry_id)
    VALUES ('delete', OLD.id, OLD.arabic_unvoweled, OLD.transliteration, OLD.id);
    INSERT INTO entries_arabic_fts (rowid, arabic_unvoweled, transliteration, entry_id)
    VALUES (NEW.id, NEW.arabic_unvoweled, NEW.transliteration, NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS trg_entries_english_fts_insert
    AFTER INSERT ON entries
BEGIN
    INSERT INTO entries_english_fts (rowid, definition, grammar_notes, entry_id)
    VALUES (NEW.id, NEW.definition, NEW.grammar_notes, NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS trg_entries_english_fts_delete
    AFTER DELETE ON entries
BEGIN
    INSERT INTO entries_english_fts (entries_english_fts, rowid, definition, grammar_notes, entry_id)
    VALUES ('delete', OLD.id, OLD.definition, OLD.grammar_notes, OLD.id);
END;

CREATE TRIGGER IF NOT EXISTS trg_entries_english_fts_update
    AFTER UPDATE ON entries
BEGIN
    INSERT INTO entries_english_fts (entries_english_fts, rowid, definition, grammar_notes, entry_id)
    VALUES ('delete', OLD.id, OLD.definition, OLD.grammar_notes, OLD.id);
    INSERT INTO entries_english_fts (rowid, definition, grammar_notes, entry_id)
    VALUES (NEW.id, NEW.definition, NEW.grammar_notes, NEW.id);
END;

-- ---------------------------------------------------------------------------
-- schema_version
-- Simple version table so import scripts can gate on schema compatibility.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version) VALUES (1);
