"""
scripts/probe_pdf.py
--------------------
Pre-flight check: sample the first N pages of the PDF and determine whether
the file contains selectable (embedded) text or is a scanned image.

This MUST be run before starting the full extraction pipeline. The entire
extract.py / parser.py approach depends on embedded text. If the PDF is a
scan, you will need to run an OCR pre-processing step (Tesseract with the
Arabic language pack) before the pipeline will produce useful output.

What it checks:
  1. Character count per page — a page with < MIN_CHARS_FOR_TEXT characters
     of actual text is likely a rasterised scan.
  2. Word count and Arabic character ratio — confirms text is actually Arabic,
     not just stray Latin page numbers/headers.
  3. Font inventory — lists the unique font names and sizes found, which helps
     calibrate ROOT_FONT_SIZE_MIN and DERIVED_FONT_SIZE_MIN in parser.py.
  4. Overall verdict: SELECTABLE | MIXED | SCANNED

Verdict thresholds:
  - If ≥ 80% of sampled pages have selectable Arabic text → SELECTABLE
  - If 20–79% of sampled pages have selectable Arabic text → MIXED (investigate)
  - If < 20% of sampled pages have selectable Arabic text → SCANNED

Usage:
  python -m scripts.probe_pdf --pdf data/hans_wehr.pdf
  python -m scripts.probe_pdf --pdf data/hans_wehr.pdf --pages 20
  python -m scripts.probe_pdf --pdf data/hans_wehr.pdf --json-out probe_result.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Check whether the PDF has selectable text or is a scanned image.")
console = Console()

# A page with fewer than this many non-whitespace characters is considered image-only.
MIN_CHARS_FOR_TEXT = 100

# Minimum fraction of characters that must be Arabic for the page to count as
# "Arabic text found" (filters out pages that are entirely headers/footers).
MIN_ARABIC_FRACTION = 0.20

# Arabic Unicode block range
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _arabic_fraction(text: str) -> float:
    stripped = text.replace(" ", "").replace("\n", "")
    if not stripped:
        return 0.0
    arabic_chars = len(_ARABIC_RE.findall(stripped))
    return arabic_chars / len(stripped)


def _analyse_page(page: fitz.Page) -> dict:
    """Extract key signals from one page."""
    text = page.get_text("text")
    char_count = len(text.replace(" ", "").replace("\n", ""))
    arabic_frac = _arabic_fraction(text)
    has_selectable_text = char_count >= MIN_CHARS_FOR_TEXT and arabic_frac >= MIN_ARABIC_FRACTION

    # Collect font names and sizes from span metadata
    fonts: list[dict] = []
    raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    font_counter: Counter = Counter()
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    key = (span.get("font", "?"), round(span.get("size", 0), 1))
                    font_counter[key] += 1

    # Top-5 font+size combos by frequency
    top_fonts = [
        {"font": f, "size": s, "span_count": n}
        for (f, s), n in font_counter.most_common(5)
    ]

    return {
        "page_number": page.number + 1,
        "char_count": char_count,
        "arabic_fraction": round(arabic_frac, 3),
        "has_selectable_text": has_selectable_text,
        "top_fonts": top_fonts,
    }


def _print_font_calibration_hint(all_results: list[dict]) -> None:
    """Aggregate font stats across pages and print hints for parser.py thresholds."""
    font_totals: Counter = Counter()
    for page_result in all_results:
        for f in page_result["top_fonts"]:
            key = (f["font"], f["size"])
            font_totals[key] += f["span_count"]

    console.print("\n[bold]Font inventory (top 20 by span count across sampled pages):[/bold]")
    table = Table(show_header=True)
    table.add_column("Font name", style="cyan")
    table.add_column("Size (pt)", style="magenta")
    table.add_column("Total spans", style="white")
    table.add_column("Likely role", style="yellow")

    for (font_name, size), count in font_totals.most_common(20):
        # Heuristic: guess the role based on size and whether name contains "Bold"
        is_bold = "bold" in font_name.lower() or "Bold" in font_name
        if is_bold and size >= 10.5:
            role = "→ ROOT headword candidate"
        elif is_bold and 8.5 <= size < 10.5:
            role = "→ derived form candidate"
        elif not is_bold and size >= 8.0:
            role = "definition / body text"
        else:
            role = "small text / footnote"
        table.add_row(font_name, str(size), str(count), role)

    console.print(table)
    console.print(
        "\n[dim]Use these size ranges to calibrate [bold]ROOT_FONT_SIZE_MIN[/bold] and "
        "[bold]DERIVED_FONT_SIZE_MIN[/bold] in src/pipeline/parser.py.[/dim]"
    )


@app.command()
def main(
    pdf: Path = typer.Option(
        Path("data/hans_wehr.pdf"),
        "--pdf", "-p",
        help="Path to the Hans Wehr PDF.",
    ),
    pages: int = typer.Option(
        10,
        "--pages",
        help="Number of pages to sample (from the beginning of the file).",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        help="Write full probe results to this JSON file.",
    ),
) -> None:
    """Check whether the PDF contains selectable text or scanned images.

    Run this before starting the extraction pipeline. If the verdict is
    SCANNED, you need to run OCR (Tesseract with Arabic support) first.
    """
    if not pdf.exists():
        console.print(f"[bold red]Error:[/bold red] PDF not found: {pdf}")
        raise typer.Exit(2)

    doc = fitz.open(str(pdf))
    total_pages = doc.page_count
    sample_size = min(pages, total_pages)

    console.print(
        f"Probing [bold]{pdf}[/bold] — {total_pages} total pages, "
        f"sampling first {sample_size}."
    )

    results = []
    for i in range(sample_size):
        page = doc[i]
        result = _analyse_page(page)
        results.append(result)

    doc.close()

    # Summary table
    table = Table(title=f"Page Probe Results (first {sample_size} pages)", show_header=True)
    table.add_column("Page", style="dim", width=5)
    table.add_column("Chars", style="white", width=8)
    table.add_column("Arabic %", style="cyan", width=10)
    table.add_column("Selectable?", style="white", width=12)

    for r in results:
        selectable_str = "[green]YES[/green]" if r["has_selectable_text"] else "[bold red]NO[/bold red]"
        table.add_row(
            str(r["page_number"]),
            str(r["char_count"]),
            f"{r['arabic_fraction'] * 100:.1f}%",
            selectable_str,
        )

    console.print(table)

    # Verdict
    selectable_count = sum(1 for r in results if r["has_selectable_text"])
    selectable_pct = selectable_count / sample_size

    if selectable_pct >= 0.80:
        verdict = "SELECTABLE"
        color = "green"
        advice = (
            "The PDF contains embedded Arabic text. "
            "You can proceed with the extraction pipeline:\n\n"
            "  uv run hans-extract --dry-run"
        )
    elif selectable_pct >= 0.20:
        verdict = "MIXED"
        color = "yellow"
        advice = (
            f"{selectable_count}/{sample_size} sampled pages have selectable text. "
            "Some pages may be scanned images. The pipeline will emit "
            "confidence=0.0 stub entries for image-only pages — review "
            "data/failed_entries.jsonl after import.\n\n"
            "You may still proceed, but expect gaps in the output."
        )
    else:
        verdict = "SCANNED"
        color = "red"
        advice = (
            "The PDF appears to be a scanned image with no embedded text.\n\n"
            "You need to run OCR before using the extraction pipeline:\n"
            "  1. Install Tesseract with the Arabic language pack (tesseract-ocr-ara)\n"
            "  2. Use ocrmypdf or a similar tool to produce a text-layer PDF:\n"
            "       ocrmypdf -l ara --force-ocr data/hans_wehr.pdf data/hans_wehr_ocr.pdf\n"
            "  3. Re-run this probe on the OCR output before extracting."
        )

    console.print(
        Panel(
            f"[bold {color}]Verdict: {verdict}[/bold {color}]\n\n{advice}",
            title="PDF Text Detection Result",
            border_style=color,
        )
    )

    _print_font_calibration_hint(results)

    if json_out:
        output = {
            "pdf": str(pdf),
            "total_pages": total_pages,
            "pages_sampled": sample_size,
            "selectable_count": selectable_count,
            "selectable_pct": round(selectable_pct, 3),
            "verdict": verdict,
            "page_results": results,
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"\nFull probe results written to {json_out}")

    if verdict == "SCANNED":
        sys.exit(1)


if __name__ == "__main__":
    app()
