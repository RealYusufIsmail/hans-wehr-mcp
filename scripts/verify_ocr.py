"""
scripts/verify_ocr.py
---------------------
OCR verification layer: cross-check Vision OCR output against Tesseract via
OCRmyPDF to catch pages where one engine made systematic errors.

How it works
------------
1. Run OCRmyPDF (Tesseract + Arabic model) on the source PDF once, producing
   a searchable PDF at data/hans_wehr_tesseract.pdf.
2. For each page_NNNN.json that was produced by scripts/ocr_pages.py (Vision),
   extract the corresponding Tesseract text from the searchable PDF via PyMuPDF.
3. Compute character trigram Jaccard similarity between the two engines' text.
4. Update the page JSON:
     - Add "tesseract_text" (raw Tesseract output, stripped of whitespace noise)
     - Add "ocr_agreement" (0.0 – 1.0)
     - If ocr_agreement < LOW_AGREEMENT_THRESHOLD, reduce every span's
       ocr_confidence by CONFIDENCE_PENALTY (clamped to 0.0).
5. Print a per-page report, sorted by agreement ascending, so the worst pages
   are immediately visible.

Pages with low agreement are flagged; when hans-parse runs on these JSONs the
lower span confidences will route more entries through the LLM refinement pass
(hans-refine --local or hans-refine), which is exactly what you want.

Requirements
------------
  OCRmyPDF:   pip install ocrmypdf    (or uv pip install hans-wehr-mcp[pipeline-local])
  Tesseract:  brew install tesseract tesseract-lang   # macOS
              apt install tesseract-ocr tesseract-ocr-ara  # Ubuntu/Debian
  Verify:     tesseract --list-langs | grep ara

When to run
-----------
  After:   scripts/ocr_pages.py  (Vision OCR → data/raw/)
  Before:  hans-parse

Skip this step if your PDF has selectable text (hans-probe reported SELECTABLE)
— it applies only to the scanned/OCR'd path.

Usage
-----
  uv run python scripts/verify_ocr.py --pdf data/hans_wehr.pdf
  uv run python scripts/verify_ocr.py --pdf data/hans_wehr.pdf --ocr-dir data/raw
  uv run python scripts/verify_ocr.py --pdf data/hans_wehr.pdf --dry-run
  uv run python scripts/verify_ocr.py --pdf data/hans_wehr.pdf --start 40 --end 60
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Cross-check Vision OCR against Tesseract/OCRmyPDF.")
console = Console()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Pages below this agreement score have span confidences penalised.
LOW_AGREEMENT_THRESHOLD = 0.40

# How much to subtract from ocr_confidence on low-agreement pages.
CONFIDENCE_PENALTY = 0.20

# Agreement below this will be highlighted red in the report.
WARN_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _require_ocrmypdf() -> None:
    """Fail early if ocrmypdf is not importable."""
    try:
        import ocrmypdf  # noqa: F401
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] ocrmypdf is not installed.\n"
            "Run:  uv pip install hans-wehr-mcp[pipeline-local]\n"
            "Or:   pip install ocrmypdf"
        )
        raise typer.Exit(2)


def _require_tesseract_arabic() -> None:
    """Fail early if tesseract is missing or lacks the Arabic data file."""
    result = subprocess.run(
        ["tesseract", "--list-langs"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print(
            "[bold red]Error:[/bold red] tesseract not found on PATH.\n"
            "macOS:   brew install tesseract tesseract-lang\n"
            "Ubuntu:  apt install tesseract-ocr tesseract-ocr-ara"
        )
        raise typer.Exit(2)

    langs = result.stdout + result.stderr
    if "ara" not in langs:
        console.print(
            "[bold red]Error:[/bold red] Tesseract Arabic language data not installed.\n"
            "macOS:   brew install tesseract-lang\n"
            "Ubuntu:  apt install tesseract-ocr-ara\n"
            "Manual:  https://github.com/tesseract-ocr/tessdata"
        )
        raise typer.Exit(2)


# ---------------------------------------------------------------------------
# OCRmyPDF runner
# ---------------------------------------------------------------------------

def run_ocrmypdf(
    pdf_path: Path,
    out_pdf_path: Path,
    start_page: int,
    end_page: int | None,
) -> None:
    """Run OCRmyPDF on the source PDF and write a searchable PDF.

    Uses the Python API rather than subprocess so stderr is captured cleanly.
    Existing text layers are overridden (--force-ocr) so we get Tesseract output
    even on PDFs that already have a partial text layer.
    """
    import ocrmypdf

    pages_arg = None
    if start_page > 1 or end_page is not None:
        end = end_page or ""
        pages_arg = f"{start_page}-{end}" if end else f"{start_page}-"

    console.print(
        f"[cyan]OCRmyPDF:[/cyan] running Tesseract (Arabic) on [bold]{pdf_path.name}[/bold] "
        f"→ [bold]{out_pdf_path.name}[/bold] …"
    )

    kwargs: dict = {
        "input_file": str(pdf_path),
        "output_file": str(out_pdf_path),
        "language": ["ara"],
        "force_ocr": True,
        "progress_bar": False,
        "optimize": 0,     # skip optimisation — we only need the text layer
    }
    if pages_arg:
        kwargs["pages"] = pages_arg

    result = ocrmypdf.ocr(**kwargs)
    if result != ocrmypdf.ExitCode.ok:
        console.print(f"[bold red]OCRmyPDF exited with code {result}[/bold red]")
        raise typer.Exit(int(result))

    console.print("[green]OCRmyPDF complete.[/green]")


# ---------------------------------------------------------------------------
# Text extraction from searchable PDF
# ---------------------------------------------------------------------------

def _extract_tesseract_text(doc: fitz.Document, page_number: int) -> str:
    """Extract plain text from one page of the OCRmyPDF output PDF."""
    try:
        page = doc[page_number - 1]  # 1-indexed
        return page.get_text("text").strip()
    except Exception as exc:
        log.warning("Could not extract text from tesseract PDF page %d: %s", page_number, exc)
        return ""


# ---------------------------------------------------------------------------
# Agreement metric
# ---------------------------------------------------------------------------

def _extract_vision_text(page_data: dict) -> str:
    """Flatten all span texts from a Vision OCR page JSON into one string."""
    parts = []
    for block in page_data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    parts.append(t)
    return " ".join(parts)


def _trigram_jaccard(text_a: str, text_b: str) -> float:
    """Character trigram Jaccard similarity — language-agnostic, order-tolerant.

    Strips whitespace before computing so word-order differences between
    engines don't artificially inflate divergence.
    """
    a = text_a.replace(" ", "").replace("\n", "")
    b = text_b.replace(" ", "").replace("\n", "")

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def _trigrams(s: str) -> set[str]:
        return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

    ta, tb = _trigrams(a), _trigrams(b)
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# JSON update
# ---------------------------------------------------------------------------

def _apply_verification(page_data: dict, tesseract_text: str, agreement: float) -> dict:
    """Return a copy of page_data with verification fields added.

    If agreement is below LOW_AGREEMENT_THRESHOLD, reduce every span's
    ocr_confidence by CONFIDENCE_PENALTY (floor 0.0).
    """
    page_data = dict(page_data)
    page_data["tesseract_text"] = tesseract_text
    page_data["ocr_agreement"] = round(agreement, 4)

    if agreement < LOW_AGREEMENT_THRESHOLD:
        penalty = CONFIDENCE_PENALTY
        for block in page_data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    current = span.get("ocr_confidence", 1.0)
                    span["ocr_confidence"] = round(max(0.0, current - penalty), 4)

        page_data.setdefault("ocr_warnings", []).append(
            f"low_agreement:{agreement:.2f}:penalised_confidence_by_{penalty}"
        )

    return page_data


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(results: list[tuple[int, float, str]]) -> None:
    """Print a summary table sorted by agreement ascending."""
    table = Table(title="OCR Verification Report", show_header=True)
    table.add_column("Page", style="cyan", justify="right")
    table.add_column("Agreement", justify="right")
    table.add_column("Status")

    # Sort by agreement (worst first)
    for page_number, agreement, status in sorted(results, key=lambda r: r[1]):
        colour = "green" if agreement >= LOW_AGREEMENT_THRESHOLD else (
            "yellow" if agreement >= WARN_THRESHOLD else "red"
        )
        table.add_row(
            str(page_number),
            f"[{colour}]{agreement:.1%}[/{colour}]",
            status,
        )

    console.print(table)

    if results:
        avg = sum(r[1] for r in results) / len(results)
        low = sum(1 for r in results if r[1] < LOW_AGREEMENT_THRESHOLD)
        console.print(
            f"\nPages verified: {len(results)}  |  "
            f"Average agreement: {avg:.1%}  |  "
            f"Low-agreement pages (penalised): [yellow]{low}[/yellow]"
        )


# ---------------------------------------------------------------------------
# Main verification loop
# ---------------------------------------------------------------------------

def verify_pages(
    pdf_path: Path,
    ocr_dir: Path,
    start_page: int,
    end_page: int | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    # Discover page JSONs that came from Vision OCR
    page_jsons = sorted(ocr_dir.glob("page_*.json"))
    if not page_jsons:
        console.print(f"[yellow]No page_NNNN.json files found in {ocr_dir}[/yellow]")
        console.print("Run scripts/ocr_pages.py first.")
        raise typer.Exit(1)

    # Filter to vision-only pages and requested range
    vision_pages: list[tuple[int, Path]] = []
    for p in page_jsons:
        try:
            page_number = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if page_number < start_page:
            continue
        if end_page is not None and page_number > end_page:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("source") == "vision_ocr":
            vision_pages.append((page_number, p))

    if not vision_pages:
        console.print(
            f"[yellow]No Vision OCR pages found in range {start_page}–{end_page or '∞'}.[/yellow]\n"
            "This script only verifies pages produced by ocr_pages.py (source=vision_ocr)."
        )
        raise typer.Exit(0)

    console.print(
        f"Found [cyan]{len(vision_pages)}[/cyan] Vision OCR pages to verify "
        f"(range {start_page}–{end_page or 'end'})."
    )

    if dry_run:
        console.print("[yellow]Dry-run:[/yellow] would verify the above pages. No files written.")
        return

    # Check already-verified pages (have ocr_agreement field)
    if not overwrite:
        already_done = [(n, p) for n, p in vision_pages
                        if json.loads(p.read_text(encoding="utf-8")).get("ocr_agreement") is not None]
        if already_done:
            console.print(
                f"[dim]{len(already_done)} pages already verified (use --overwrite to redo).[/dim]"
            )
        vision_pages = [(n, p) for n, p in vision_pages
                        if json.loads(p.read_text(encoding="utf-8")).get("ocr_agreement") is None]
        if not vision_pages:
            console.print("[green]All pages already verified.[/green]")
            return

    first_page = min(n for n, _ in vision_pages)
    last_page = max(n for n, _ in vision_pages)

    # Run OCRmyPDF once covering the required page range
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tess_pdf_path = Path(tmp.name)

    try:
        run_ocrmypdf(pdf_path, tess_pdf_path, first_page, last_page)

        tess_doc = fitz.open(str(tess_pdf_path))

        results: list[tuple[int, float, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Verifying pages", total=len(vision_pages))

            for page_number, json_path in vision_pages:
                page_data = json.loads(json_path.read_text(encoding="utf-8"))

                vision_text = _extract_vision_text(page_data)
                # tess_doc is indexed from 0 over the original PDF pages
                tess_text = _extract_tesseract_text(tess_doc, page_number)

                agreement = _trigram_jaccard(vision_text, tess_text)
                penalised = agreement < LOW_AGREEMENT_THRESHOLD

                updated = _apply_verification(page_data, tess_text, agreement)
                json_path.write_text(
                    json.dumps(updated, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                status = "[red]PENALISED[/red]" if penalised else "[green]OK[/green]"
                results.append((page_number, agreement, status))
                log.debug("Page %d: agreement=%.1f%% %s", page_number, agreement * 100,
                          "PENALISED" if penalised else "OK")

                progress.advance(task)

        tess_doc.close()

    finally:
        tess_pdf_path.unlink(missing_ok=True)

    _print_report(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    pdf: Path = typer.Option(
        Path("data/hans_wehr.pdf"),
        "--pdf", "-p",
        help="Source PDF (same one used for ocr_pages.py).",
    ),
    ocr_dir: Path = typer.Option(
        Path("data/raw"),
        "--ocr-dir",
        help="Directory containing page_NNNN.json files from ocr_pages.py.",
    ),
    start: int = typer.Option(1, "--start", help="First page to verify (1-indexed)."),
    end: int | None = typer.Option(None, "--end", help="Last page to verify (inclusive)."),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="Re-verify pages that already have an ocr_agreement field.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show which pages would be verified without running OCRmyPDF.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Cross-check Vision OCR output against Tesseract (OCRmyPDF).

    Run this after ocr_pages.py and before hans-parse.  Pages where the two
    engines disagree have their span ocr_confidence values reduced, which
    routes them through LLM refinement automatically.

    \b
    Requires:
      tesseract with Arabic data
        macOS:  brew install tesseract tesseract-lang
        Ubuntu: apt install tesseract-ocr tesseract-ocr-ara

    \b
    Typical workflow:
      uv run python scripts/ocr_pages.py --pdf data/hans_wehr.pdf
      uv run python scripts/verify_ocr.py --pdf data/hans_wehr.pdf
      uv run hans-parse
      uv run hans-refine --local
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    _require_ocrmypdf()
    _require_tesseract_arabic()

    if not pdf.exists():
        console.print(f"[bold red]Error:[/bold red] PDF not found: {pdf}")
        raise typer.Exit(2)

    if not ocr_dir.exists():
        console.print(f"[bold red]Error:[/bold red] OCR directory not found: {ocr_dir}")
        console.print("Run scripts/ocr_pages.py first.")
        raise typer.Exit(2)

    verify_pages(pdf, ocr_dir, start, end, overwrite, dry_run)


if __name__ == "__main__":
    app()
