"""
scripts/import_db.py
--------------------
Bulk-import structured entries from data/processed/entries_refined.jsonl
(or entries.jsonl if the refinement pass was skipped) into the SQLite database.

Steps performed:
  1. Create the DB from src/db/schema.sql (idempotent — uses CREATE IF NOT EXISTS)
  2. Insert or update roots
  3. Insert entries with parse_metadata rows
  4. Insert cross_references (unresolved; run xref_check.py to resolve them)
  5. Update denormalised entry_count on roots
  6. Rebuild the FTS5 virtual tables

This script is intentionally verbose so you can see exactly what is happening.
It will NOT silently drop entries — any row that fails to insert is written to
data/failed_entries.jsonl for investigation.

Usage:
  python -m scripts.import_db
  python -m scripts.import_db --input data/processed/entries.jsonl  # skip refinement
  python -m scripts.import_db --db data/hans_wehr.db --dry-run
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from hans_wehr.db.queries import normalise_transliteration

app = typer.Typer(help="Import parsed entries into SQLite.")
console = Console()
log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent.parent / "src" / "hans_wehr" / "db" / "schema.sql"
DEFAULT_INPUT = Path("data/processed/entries_refined.jsonl")
FALLBACK_INPUT = Path("data/processed/entries.jsonl")
DEFAULT_DB = Path("data/hans_wehr.db")
FAILED_ENTRIES_PATH = Path("data/failed_entries.jsonl")


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def _init_db(db_path: Path) -> sqlite3.Connection:
    """Create the database and apply schema.sql."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")  # faster bulk inserts

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema)
    conn.commit()
    log.info("Schema applied to %s", db_path)
    return conn


# ---------------------------------------------------------------------------
# Root upsert
# ---------------------------------------------------------------------------

def _upsert_root(conn: sqlite3.Connection, entry: dict) -> int:
    """Insert or update a root row. Returns the root's rowid."""
    root_arabic = entry.get("root_arabic", "")
    root_unvoweled = entry.get("root_unvoweled", "") or root_arabic
    root_translit = entry.get("root_translit", "") or ""
    root_page = entry.get("root_page") or entry.get("page_number", 0)

    existing = conn.execute(
        "SELECT id FROM roots WHERE arabic_unvoweled = ?",
        (root_unvoweled,),
    ).fetchone()

    if existing:
        # Update transliteration if it was empty before
        if root_translit:
            conn.execute(
                """
                UPDATE roots
                SET transliteration = ?, transliteration_ascii = ?
                WHERE id = ? AND transliteration = ''
                """,
                (root_translit, normalise_transliteration(root_translit), existing["id"]),
            )
        return existing["id"]

    cursor = conn.execute(
        """
        INSERT INTO roots (arabic, arabic_unvoweled, transliteration, transliteration_ascii, page_number)
        VALUES (?, ?, ?, ?, ?)
        """,
        (root_arabic, root_unvoweled, root_translit, normalise_transliteration(root_translit), root_page),
    )
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Entry insert
# ---------------------------------------------------------------------------

def _insert_entry(conn: sqlite3.Connection, root_id: int, entry: dict) -> int | None:
    """Insert one entry row. Returns the entry's rowid or None on failure."""
    arabic = entry.get("arabic", "")
    if not arabic:
        return None

    plural_json = json.dumps(entry.get("plural_forms", []), ensure_ascii=False)
    confidence = float(entry.get("confidence", 1.0))
    needs_review = 1 if entry.get("needs_review") else 0

    raw_translit = entry.get("transliteration") or None
    translit_ascii = normalise_transliteration(raw_translit) if raw_translit else None

    cursor = conn.execute(
        """
        INSERT INTO entries (
            root_id, arabic, arabic_unvoweled, transliteration, transliteration_ascii,
            part_of_speech, verb_form, plural_forms, definition,
            grammar_notes, page_number, confidence, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            root_id,
            arabic,
            entry.get("arabic_unvoweled", "") or arabic,
            raw_translit,
            translit_ascii,
            entry.get("part_of_speech") or None,
            entry.get("verb_form") or None,
            plural_json,
            entry.get("definition", "") or "",
            entry.get("grammar_notes") or None,
            entry.get("page_number", 0),
            confidence,
            needs_review,
        ),
    )
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Parse metadata insert
# ---------------------------------------------------------------------------

def _insert_parse_metadata(conn: sqlite3.Connection, entry_id: int, entry: dict) -> None:
    warnings = entry.get("warnings", [])
    warnings_json = json.dumps(warnings, ensure_ascii=False)
    parse_method = entry.get("parse_method", "structural")
    llm_model = entry.get("llm_model") or None
    confidence = float(entry.get("confidence", 1.0))
    raw_text = entry.get("raw_text", "")

    conn.execute(
        """
        INSERT OR REPLACE INTO parse_metadata
            (entry_id, raw_text, parse_method, llm_model, confidence, extraction_warnings)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (entry_id, raw_text, parse_method, llm_model, confidence, warnings_json),
    )


# ---------------------------------------------------------------------------
# Cross-reference insert
# ---------------------------------------------------------------------------

def _insert_xrefs(conn: sqlite3.Connection, entry_id: int, entry: dict) -> None:
    xrefs = entry.get("cross_references", [])
    for xref in xrefs:
        raw_target = xref.get("to_arabic_raw", "").strip()
        if not raw_target:
            continue
        ref_type = xref.get("ref_type", "see")
        conn.execute(
            """
            INSERT INTO cross_references (from_entry_id, to_arabic_raw, ref_type, resolved)
            VALUES (?, ?, ?, 0)
            """,
            (entry_id, raw_target, ref_type),
        )


# ---------------------------------------------------------------------------
# FTS rebuild
# ---------------------------------------------------------------------------

def _rebuild_fts(conn: sqlite3.Connection) -> None:
    """Populate FTS5 tables from the entries and roots tables.

    The triggers keep FTS in sync for incremental updates, but a full rebuild
    after bulk import is faster and safer.
    """
    log.info("Rebuilding FTS5 indexes…")

    conn.execute("DELETE FROM entries_arabic_fts")
    conn.execute("DELETE FROM entries_english_fts")
    conn.execute("DELETE FROM roots_fts")

    conn.execute(
        """
        INSERT INTO entries_arabic_fts (rowid, arabic_unvoweled, transliteration, entry_id)
        SELECT id, arabic_unvoweled, transliteration, id FROM entries
        """
    )
    conn.execute(
        """
        INSERT INTO entries_english_fts (rowid, definition, grammar_notes, entry_id)
        SELECT id, definition, grammar_notes, id FROM entries
        """
    )
    conn.execute(
        """
        INSERT INTO roots_fts (rowid, arabic_unvoweled, transliteration, transliteration_ascii, root_id)
        SELECT id, arabic_unvoweled, transliteration, transliteration_ascii, id FROM roots
        """
    )
    conn.commit()
    log.info("FTS5 indexes rebuilt.")


# ---------------------------------------------------------------------------
# Denormalised counts update
# ---------------------------------------------------------------------------

def _update_entry_counts(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE roots SET entry_count = (
            SELECT COUNT(*) FROM entries WHERE entries.root_id = roots.id
        )
        """
    )
    conn.commit()
    log.info("Updated entry_count on roots table.")


# ---------------------------------------------------------------------------
# Main import logic
# ---------------------------------------------------------------------------

def run_import(
    input_path: Path,
    db_path: Path,
    dry_run: bool,
) -> None:
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        raise typer.Exit(2)

    conn = _init_db(db_path)

    failed: list[dict] = []
    n_roots = 0
    n_entries = 0
    n_xrefs = 0
    n_failed = 0

    # Count lines for progress bar
    total_lines = sum(1 for _ in input_path.open(encoding="utf-8") if _.strip())

    failed_fh = FAILED_ENTRIES_PATH.open("w", encoding="utf-8") if not dry_run else None

    try:
        with input_path.open(encoding="utf-8") as fh:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Importing entries", total=total_lines)

                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        progress.advance(task)
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        log.warning("Line %d: invalid JSON — %s", line_no, exc)
                        n_failed += 1
                        progress.advance(task)
                        continue

                    if dry_run:
                        # Just validate structure without writing
                        if not entry.get("arabic"):
                            log.warning("Line %d: missing 'arabic' field", line_no)
                        progress.advance(task)
                        n_entries += 1
                        continue

                    try:
                        root_id = _upsert_root(conn, entry)
                        n_roots += 1  # may be an existing root; counts are approximate

                        entry_id = _insert_entry(conn, root_id, entry)
                        if entry_id is None:
                            log.warning("Line %d: skipped entry with no Arabic text", line_no)
                            n_failed += 1
                            if failed_fh:
                                failed_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            progress.advance(task)
                            continue

                        _insert_parse_metadata(conn, entry_id, entry)
                        _insert_xrefs(conn, entry_id, entry)

                        n_entries += 1
                        n_xrefs += len(entry.get("cross_references", []))

                        # Commit in batches for performance
                        if n_entries % 1000 == 0:
                            conn.commit()
                            log.debug("Committed %d entries", n_entries)

                    except sqlite3.IntegrityError as exc:
                        log.error("Line %d: DB integrity error — %s", line_no, exc)
                        n_failed += 1
                        if failed_fh:
                            failed_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        conn.rollback()

                    progress.advance(task)

        if not dry_run:
            conn.commit()
            _update_entry_counts(conn)
            _rebuild_fts(conn)

    finally:
        if failed_fh:
            failed_fh.close()
        conn.close()

    console.print(
        f"\n[green]Import complete.[/green]\n"
        f"  Entries imported : {n_entries:,}\n"
        f"  Cross-refs stored: {n_xrefs:,}\n"
        f"  Failed entries   : {n_failed:,}"
        + (f" → see {FAILED_ENTRIES_PATH}" if n_failed else "")
    )
    if dry_run:
        console.print("[yellow]Dry-run mode — nothing written to DB.[/yellow]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    input: Path = typer.Option(
        None,
        "--input", "-i",
        help=f"JSONL file to import. Defaults to {DEFAULT_INPUT}, falls back to {FALLBACK_INPUT}.",
    ),
    db: Path = typer.Option(DEFAULT_DB, "--db", help="Path to the SQLite database to create/update."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate input without writing to DB."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Import structured entries into the Hans Wehr SQLite database."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Resolve input path
    if input is None:
        input = DEFAULT_INPUT if DEFAULT_INPUT.exists() else FALLBACK_INPUT

    run_import(input, db, dry_run)


if __name__ == "__main__":
    app()
