"""
src/db/queries.py
-----------------
All database read queries used by the MCP server and validation tools.

Design rules:
- Every function accepts a sqlite3.Connection (or sqlite_utils.Database) so
  tests can inject an in-memory DB without touching the filesystem.
- No query string concatenation with user input — always use parameterised
  queries to prevent SQL injection.
- Arabic input is always normalised (diacritics stripped) before querying
  the *_unvoweled columns and FTS tables.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Arabic normalisation helpers
# ---------------------------------------------------------------------------

# Unicode ranges covering Arabic tashkeel (diacritics / harakat):
#   U+064B FATHATAN … U+065F WAVY HAMZA BELOW
#   U+0670 ARABIC LETTER SUPERSCRIPT ALEF
_DIACRITIC_RE = re.compile(r"[\u064B-\u065F\u0670]")

# Latin combining diacritics (U+0300 COMBINING GRAVE … U+036F COMBINING LATIN SMALL LETTER X)
# Used to normalise Hans Wehr transliteration: ā→a, ī→i, ū→u, ḥ→h, ṭ→t, ẓ→z, etc.
_LATIN_COMBINING_RE = re.compile(r"[\u0300-\u036F]")


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel from *text*, returning the bare consonantal form."""
    return _DIACRITIC_RE.sub("", text)


def normalise_transliteration(text: str) -> str:
    """Strip Latin combining diacritics from a Hans Wehr transliteration string.

    Converts academic ALA-LC transliteration to plain ASCII so users can type
    without special characters:
        "kātaba"  → "kataba"
        "ḥusn"    → "husn"
        "ṭalab"   → "talab"
        "maʿrūf"  → "maʿruf"   (ʿ ayin is not a combining mark; kept as-is)

    Uses Unicode NFD decomposition to separate base characters from their
    combining marks, then strips the combining marks (U+0300–U+036F).
    Result is lowercased and stripped of leading/trailing whitespace.
    """
    nfd = unicodedata.normalize("NFD", text)
    return _LATIN_COMBINING_RE.sub("", nfd).lower().strip()


def normalise_arabic_query(query: str) -> str:
    """Prepare an Arabic query string for FTS or exact lookup.

    Strips diacritics and trims whitespace. Does NOT do tatweel (kashida)
    removal — that would be lossy for proper-noun lookup.
    """
    return strip_diacritics(query.strip())


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a read-optimised SQLite connection.

    Callers are responsible for closing the connection (use as context manager).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows accessible by column name
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA query_only=ON;")  # safety: this module is read-only
    return conn


# ---------------------------------------------------------------------------
# Root queries
# ---------------------------------------------------------------------------

def get_root_by_arabic(conn: sqlite3.Connection, arabic: str) -> sqlite3.Row | None:
    """Fetch a root by its Arabic text (voweled or unvoweled)."""
    unvoweled = strip_diacritics(arabic.strip())
    return conn.execute(
        "SELECT * FROM roots WHERE arabic_unvoweled = ?",
        (unvoweled,),
    ).fetchone()


def get_root_by_transliteration(conn: sqlite3.Connection, translit: str) -> sqlite3.Row | None:
    """Fetch a root by its transliteration.

    Two-step lookup:
    1. Exact case-insensitive match on the stored transliteration  (e.g. "kātaba").
    2. Normalised ASCII match on transliteration_ascii             (e.g. "kataba").

    This lets users type without special diacritic characters and still find roots.
    """
    t = translit.strip()
    row = conn.execute(
        "SELECT * FROM roots WHERE lower(transliteration) = lower(?)",
        (t,),
    ).fetchone()
    if row is not None:
        return row

    # Fallback: strip Latin combining marks (ā→a, ḥ→h, etc.) and try ascii column
    ascii_form = normalise_transliteration(t)
    return conn.execute(
        "SELECT * FROM roots WHERE transliteration_ascii = ?",
        (ascii_form,),
    ).fetchone()


def get_root_by_id(conn: sqlite3.Connection, root_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM roots WHERE id = ?", (root_id,)).fetchone()


def list_roots_by_letter(conn: sqlite3.Connection, letter: str) -> list[sqlite3.Row]:
    """Return all roots whose arabic_unvoweled starts with *letter*.

    *letter* may be a single Arabic character or a transliteration letter.
    We detect which by checking if it's in the Arabic Unicode block (U+0600–U+06FF).
    """
    letter = letter.strip()
    if letter and "\u0600" <= letter[0] <= "\u06FF":
        # Arabic script letter — strip diacritics and use prefix match
        bare = strip_diacritics(letter)
        return conn.execute(
            "SELECT * FROM roots WHERE arabic_unvoweled LIKE ? ORDER BY page_number",
            (f"{bare}%",),
        ).fetchall()
    else:
        # Assume transliteration letter
        return conn.execute(
            "SELECT * FROM roots WHERE lower(transliteration) LIKE lower(?) ORDER BY page_number",
            (f"{letter}%",),
        ).fetchall()


# ---------------------------------------------------------------------------
# Entry queries
# ---------------------------------------------------------------------------

def get_entry_by_id(conn: sqlite3.Connection, entry_id: int) -> sqlite3.Row | None:
    """Full entry record including parse metadata."""
    return conn.execute(
        """
        SELECT
            e.*,
            pm.raw_text,
            pm.parse_method,
            pm.llm_model,
            pm.extraction_warnings
        FROM entries e
        LEFT JOIN parse_metadata pm ON pm.entry_id = e.id
        WHERE e.id = ?
        """,
        (entry_id,),
    ).fetchone()


def get_entries_for_root(conn: sqlite3.Connection, root_id: int) -> list[sqlite3.Row]:
    """All entries under a root, ordered by page then rowid."""
    return conn.execute(
        """
        SELECT * FROM entries
        WHERE root_id = ?
        ORDER BY page_number, id
        """,
        (root_id,),
    ).fetchall()


def get_cross_references_for_entry(
    conn: sqlite3.Connection, entry_id: int
) -> list[sqlite3.Row]:
    """Return all cross-references originating from *entry_id*."""
    return conn.execute(
        """
        SELECT xr.*, e.arabic AS to_arabic, e.definition AS to_definition
        FROM cross_references xr
        LEFT JOIN entries e ON e.id = xr.to_entry_id
        WHERE xr.from_entry_id = ?
        """,
        (entry_id,),
    ).fetchall()


# ---------------------------------------------------------------------------
# FTS queries
# ---------------------------------------------------------------------------

def search_arabic(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Full-text search across Arabic words and transliterations.

    The query is normalised (diacritics stripped) before being passed to FTS5.
    Returns a list of dicts with entry fields plus a snippet.
    """
    bare_query = normalise_arabic_query(query)
    # FTS5 MATCH syntax: wrap in double quotes to treat as a phrase
    fts_query = f'"{bare_query}"' if " " in bare_query else bare_query
    rows = conn.execute(
        """
        SELECT
            e.id,
            e.arabic,
            e.arabic_unvoweled,
            e.transliteration,
            e.part_of_speech,
            e.definition,
            e.confidence,
            e.page_number,
            snippet(entries_arabic_fts, 0, '<b>', '</b>', '…', 10) AS match_snippet
        FROM entries_arabic_fts fts
        JOIN entries e ON e.id = fts.entry_id
        WHERE entries_arabic_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def search_english(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Full-text search across English definitions (Porter-stemmed).

    Returns a list of dicts with entry fields plus a snippet.
    """
    query = query.strip()
    fts_query = f'"{query}"' if " " in query else query
    rows = conn.execute(
        """
        SELECT
            e.id,
            e.arabic,
            e.transliteration,
            e.part_of_speech,
            e.definition,
            e.confidence,
            e.page_number,
            snippet(entries_english_fts, 0, '<b>', '</b>', '…', 20) AS match_snippet
        FROM entries_english_fts fts
        JOIN entries e ON e.id = fts.entry_id
        WHERE entries_english_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def search_transliteration(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search entries by transliteration using FTS5.

    Works with or without diacritics — the FTS tokenizer strips combining marks
    on both the stored text and the query, so "kataba" matches "kātaba".

    Uses the FTS5 column filter syntax to restrict matching to the
    transliteration column only (avoids Arabic false-positives).
    """
    q = query.strip().lower()
    # FTS5 column filter: "transliteration : <term>"
    fts_query = f'transliteration : "{q}"' if " " in q else f"transliteration : {q}"
    rows = conn.execute(
        """
        SELECT
            e.id,
            e.arabic,
            e.arabic_unvoweled,
            e.transliteration,
            e.transliteration_ascii,
            e.part_of_speech,
            e.definition,
            e.confidence,
            e.page_number,
            snippet(entries_arabic_fts, 1, '<b>', '</b>', '…', 10) AS match_snippet
        FROM entries_arabic_fts fts
        JOIN entries e ON e.id = fts.entry_id
        WHERE entries_arabic_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def search_roots_by_transliteration(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> list[sqlite3.Row]:
    """Search roots by transliteration using roots_fts.

    Handles casual input (no diacritics) because the FTS tokenizer applies
    remove_diacritics to both stored text and the query.
    """
    q = query.strip().lower()
    fts_query = f'transliteration : "{q}"' if " " in q else f"transliteration : {q}"
    rows = conn.execute(
        """
        SELECT r.*
        FROM roots_fts fts
        JOIN roots r ON r.id = fts.root_id
        WHERE roots_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()
    return rows


# ---------------------------------------------------------------------------
# Validation / reporting queries
# ---------------------------------------------------------------------------

def count_roots(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM roots").fetchone()[0]


def count_entries(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]


def count_needs_review(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM entries WHERE needs_review = 1"
    ).fetchone()[0]


def count_unresolved_xrefs(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM cross_references WHERE resolved = 0"
    ).fetchone()[0]


def get_random_entries(conn: sqlite3.Connection, n: int = 100) -> list[sqlite3.Row]:
    """Return *n* randomly sampled entries for spot-checking."""
    return conn.execute(
        "SELECT * FROM entries ORDER BY RANDOM() LIMIT ?",
        (n,),
    ).fetchall()


def get_low_confidence_entries(
    conn: sqlite3.Connection,
    threshold: float = 0.75,
    limit: int = 500,
) -> list[sqlite3.Row]:
    """Entries whose confidence score is below *threshold*, for LLM refinement."""
    return conn.execute(
        """
        SELECT e.*, pm.raw_text
        FROM entries e
        JOIN parse_metadata pm ON pm.entry_id = e.id
        WHERE e.confidence < ? AND e.needs_review = 1
        ORDER BY e.confidence ASC
        LIMIT ?
        """,
        (threshold, limit),
    ).fetchall()


def parse_plural_forms(plural_forms_json: str | None) -> list[str]:
    """Deserialise the plural_forms JSON column to a Python list."""
    if not plural_forms_json:
        return []
    try:
        return json.loads(plural_forms_json)
    except json.JSONDecodeError:
        return []
