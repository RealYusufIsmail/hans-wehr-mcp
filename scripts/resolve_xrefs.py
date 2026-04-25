"""
scripts/resolve_xrefs.py
------------------------
Post-import pipeline step: resolve the cross_references table.

Run this AFTER scripts/import_db.py.  Without it, every row in
cross_references will have resolved=0 and to_entry_id=NULL forever —
making the `get_entry` MCP tool return empty cross-reference lists.

Resolution strategy (in order):
  1. Exact match on entries.arabic_unvoweled
  2. Exact match on entries.arabic (voweled form)
  3. Prefix match on arabic_unvoweled (first 4 chars) — catches minor
     spelling variants and truncated targets

The script is idempotent: re-running it will only attempt to resolve
rows that are still unresolved (resolved=0), so it is safe to run again
after adding new entries or after the LLM refinement pass.

Pipeline position:
  hans-extract → hans-parse → hans-refine → hans-import → hans-resolve-xrefs → hans-validate

Usage:
  python -m scripts.resolve_xrefs
  python -m scripts.resolve_xrefs --db data/hans_wehr.db --dry-run
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.db.queries import strip_diacritics

app = typer.Typer(help="Resolve cross-references in the dictionary database (post-import step).")
console = Console()
log = logging.getLogger(__name__)

DEFAULT_DB = Path("data/hans_wehr.db")

# Minimum length of unvoweled prefix to use for prefix-match fallback.
# Using < 4 chars risks too many false matches (e.g. "كت" matches everything
# under the كتب root).
MIN_PREFIX_LENGTH = 4


def _resolve(conn: sqlite3.Connection, dry_run: bool) -> dict:
    """Attempt to resolve all unresolved cross-references.

    Returns a stats dict with counts and common unresolved targets.
    """
    unresolved_rows = conn.execute(
        """
        SELECT id, from_entry_id, to_arabic_raw
        FROM cross_references
        WHERE resolved = 0
        ORDER BY id
        """
    ).fetchall()

    total = len(unresolved_rows)
    if total == 0:
        return {
            "total": 0, "resolved": 0, "still_unresolved": 0,
            "resolution_rate_pct": 100.0, "top_unresolved": [],
        }

    resolved_count = 0
    still_unresolved: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Resolving {total:,} cross-references", total=total)

        for row in unresolved_rows:
            xref_id = row["id"]
            raw = (row["to_arabic_raw"] or "").strip()

            if not raw:
                still_unresolved.append("<empty>")
                progress.advance(task)
                continue

            bare = strip_diacritics(raw)
            match_id: int | None = None

            # Strategy 1: exact unvoweled match
            hit = conn.execute(
                "SELECT id FROM entries WHERE arabic_unvoweled = ? LIMIT 1",
                (bare,),
            ).fetchone()
            if hit:
                match_id = hit["id"]

            # Strategy 2: exact voweled match
            if match_id is None:
                hit = conn.execute(
                    "SELECT id FROM entries WHERE arabic = ? LIMIT 1",
                    (raw,),
                ).fetchone()
                if hit:
                    match_id = hit["id"]

            # Strategy 3: prefix match (conservative — requires MIN_PREFIX_LENGTH chars)
            if match_id is None and len(bare) >= MIN_PREFIX_LENGTH:
                hit = conn.execute(
                    "SELECT id FROM entries WHERE arabic_unvoweled LIKE ? LIMIT 1",
                    (bare[:MIN_PREFIX_LENGTH] + "%",),
                ).fetchone()
                if hit:
                    match_id = hit["id"]
                    log.debug(
                        "xref %d resolved via prefix match: %r → entry %d",
                        xref_id, raw, match_id,
                    )

            if match_id is not None:
                if not dry_run:
                    conn.execute(
                        "UPDATE cross_references SET to_entry_id = ?, resolved = 1 WHERE id = ?",
                        (match_id, xref_id),
                    )
                resolved_count += 1
            else:
                still_unresolved.append(bare)

            progress.advance(task)

    if not dry_run:
        conn.commit()

    unresolved_counts = Counter(still_unresolved)
    resolution_rate = round(resolved_count / max(total, 1) * 100, 2)

    return {
        "total": total,
        "resolved": resolved_count,
        "still_unresolved": total - resolved_count,
        "resolution_rate_pct": resolution_rate,
        "top_unresolved": unresolved_counts.most_common(20),
    }


@app.command()
def main(
    db: Path = typer.Option(DEFAULT_DB, "--db", help="Path to the SQLite database."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be resolved without writing any changes.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Resolve cross-references after import.

    This is pipeline step 5. Run it after hans-import and before hans-validate.

    Safe to run multiple times — only touches rows where resolved=0.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not db.exists():
        console.print(f"[bold red]Error:[/bold red] DB not found: {db}")
        raise typer.Exit(2)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")

    # Quick check: how many unresolved xrefs exist before we start
    unresolved_before = conn.execute(
        "SELECT COUNT(*) FROM cross_references WHERE resolved = 0"
    ).fetchone()[0]
    total_xrefs = conn.execute("SELECT COUNT(*) FROM cross_references").fetchone()[0]

    console.print(
        f"Database: {db}\n"
        f"Total cross-references: {total_xrefs:,}  |  "
        f"Already resolved: {total_xrefs - unresolved_before:,}  |  "
        f"Pending: {unresolved_before:,}"
    )

    if unresolved_before == 0:
        console.print("[green]All cross-references are already resolved.[/green]")
        conn.close()
        return

    stats = _resolve(conn, dry_run)
    conn.close()

    # Results table
    table = Table(title="Cross-Reference Resolution Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Attempted", f"{stats['total']:,}")
    table.add_row("Resolved this run", f"{stats['resolved']:,}")
    table.add_row("Still unresolved", f"{stats['still_unresolved']:,}")
    table.add_row(
        "Resolution rate",
        f"{stats['resolution_rate_pct']}%"
        + (" [dim](target ≥90%)[/dim]" if stats["resolution_rate_pct"] < 90 else " [green]✓[/green]"),
    )
    console.print(table)

    if stats["top_unresolved"]:
        console.print(
            "\n[bold]Top unresolved targets[/bold] [dim](most frequent first — "
            "these may indicate parsing errors or cross-references to roots not yet in the DB)[/dim]"
        )
        miss_table = Table(show_header=True)
        miss_table.add_column("Target (unvoweled)", style="yellow")
        miss_table.add_column("Count", style="magenta")
        for target, count in stats["top_unresolved"]:
            miss_table.add_row(target, str(count))
        console.print(miss_table)

    if dry_run:
        console.print("\n[yellow]Dry-run — no changes written to DB.[/yellow]")
    else:
        console.print(f"\n[green]Done.[/green] Updated {stats['resolved']:,} rows in {db}")

    if stats["resolution_rate_pct"] < 90:
        console.print(
            f"\n[yellow]Warning:[/yellow] resolution rate {stats['resolution_rate_pct']}% "
            f"is below the 90% target. Common causes:\n"
            f"  • The referenced root/entry hasn't been imported yet\n"
            f"  • The cross-reference text was garbled during extraction\n"
            f"  • Run the LLM refinement pass (hans-refine) to fix low-confidence entries,\n"
            f"    then re-run this script\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    app()
