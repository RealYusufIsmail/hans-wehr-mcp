"""
src/validation/root_count.py
-----------------------------
Verify that the parsed root and entry counts are within expected ranges for
the Hans Wehr 4th edition:
  - Roots: ~13,000  (tolerance ±5%)
  - Entries: ~60,000 (tolerance ±15%)
  - No root should have zero entries
  - Distribution of entries per root should look roughly normal (1–50)

NOTE on entry tolerance: different PDF sources and editions vary significantly
in how entries are split across lines and pages. ±15% is intentionally loose
for the first import pass. Tighten it (e.g. to ±5%) once you have run the
pipeline against your specific copy and know what it actually yields.

Usage:
  python -m src.validation.root_count --db data/hans_wehr.db
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.db.queries import (
    get_connection,
    count_roots,
    count_entries,
    count_needs_review,
    count_unresolved_xrefs,
)

app = typer.Typer(help="Validate root and entry counts against expected ranges.")
console = Console()

EXPECTED_ROOTS = 13_000
EXPECTED_ENTRIES = 60_000
ROOT_TOLERANCE = 0.05    # ±5%
ENTRY_TOLERANCE = 0.15   # ±15% — intentionally loose; tighten after first full run


def _check(label: str, actual: int, expected: int, tolerance: float) -> tuple[bool, str]:
    lo = int(expected * (1 - tolerance))
    hi = int(expected * (1 + tolerance))
    ok = lo <= actual <= hi
    status = "[green]PASS[/green]" if ok else "[bold red]FAIL[/bold red]"
    msg = f"{actual:,} (expected {lo:,}–{hi:,})"
    return ok, f"{status}  {label}: {msg}"


@app.command()
def main(
    db: Path = typer.Option(Path("data/hans_wehr.db"), "--db", help="Path to the SQLite DB."),
) -> None:
    """Check that root/entry counts are within expected ranges for Hans Wehr 4th ed."""
    if not db.exists():
        console.print(f"[bold red]Error:[/bold red] DB not found: {db}")
        raise typer.Exit(2)

    with get_connection(str(db)) as conn:
        n_roots = count_roots(conn)
        n_entries = count_entries(conn)
        n_review = count_needs_review(conn)
        n_unresolved_xrefs = count_unresolved_xrefs(conn)

        # Check for roots with zero entries
        zero_entry_roots = conn.execute(
            "SELECT COUNT(*) FROM roots WHERE entry_count = 0"
        ).fetchone()[0]

        # Entries per root distribution (min, max, avg)
        dist = conn.execute(
            "SELECT MIN(entry_count), MAX(entry_count), AVG(entry_count) FROM roots WHERE entry_count > 0"
        ).fetchone()
        min_entries, max_entries, avg_entries = dist if dist else (0, 0, 0)

    all_passed = True

    table = Table(title="Root/Entry Count Validation", show_header=True, header_style="bold")
    table.add_column("Check", style="white")
    table.add_column("Result", style="white")

    root_ok, root_msg = _check("Root count", n_roots, EXPECTED_ROOTS, ROOT_TOLERANCE)
    entry_ok, entry_msg = _check("Entry count", n_entries, EXPECTED_ENTRIES, ENTRY_TOLERANCE)

    table.add_row("Root count", root_msg)
    table.add_row("Entry count", entry_msg)

    zero_ok = zero_entry_roots == 0
    table.add_row(
        "Roots with zero entries",
        f"[green]PASS[/green]  0" if zero_ok else f"[bold red]FAIL[/bold red]  {zero_entry_roots:,}",
    )

    table.add_row("Needs review", f"{n_review:,} ({100 * n_review / max(n_entries, 1):.1f}%)")
    table.add_row("Unresolved cross-refs", f"{n_unresolved_xrefs:,}")
    table.add_row("Entries per root (min/avg/max)", f"{min_entries} / {avg_entries:.1f} / {max_entries}")

    console.print(table)

    all_passed = root_ok and entry_ok and zero_ok
    if all_passed:
        console.print("\n[bold green]All validation checks passed.[/bold green]")
    else:
        console.print("\n[bold red]One or more validation checks failed.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
