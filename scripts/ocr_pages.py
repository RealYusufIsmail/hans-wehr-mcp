"""
scripts/ocr_pages.py
--------------------
On-device OCR for scanned PDF pages using the macOS Vision framework.

Use this instead of hans-extract when hans-probe reports SCANNED — i.e. the
PDF has no embedded text layer.  Runs entirely on Apple Silicon's Neural
Engine; no API key, no network, no cost.

Output: the same data/raw/page_NNNN.json format that hans-parse expects, so
the rest of the pipeline is unchanged.  Spans produced by OCR carry an
"ocr_confidence" field and have is_bold/is_italic set to False (Vision does
not report font weight).  The structural parser will therefore assign lower
confidence scores, routing more entries through the --local refinement pass.

Requirements:
  macOS 13+  (Ventura) — VNRecognizeTextRequest Arabic support
  Apple Silicon or Intel Mac with Neural Engine
  uv pip install hans-wehr-mcp[pipeline-local]

Usage:
  uv run python scripts/ocr_pages.py --pdf data/hans_wehr.pdf
  uv run python scripts/ocr_pages.py --pdf data/hans_wehr.pdf --dry-run
  uv run python scripts/ocr_pages.py --pdf data/hans_wehr.pdf --start 40 --end 60
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF — renders pages to PNG for Vision input
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

app = typer.Typer(help="On-device OCR for scanned Arabic PDFs using macOS Vision.")
console = Console()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# macOS guard — fail early with a useful message on non-Mac systems
# ---------------------------------------------------------------------------

def _require_macos() -> None:
    if platform.system() != "Darwin":
        console.print(
            "[bold red]Error:[/bold red] ocr_pages.py requires macOS.\n"
            "On Linux/Windows use Tesseract instead:\n"
            "  pip install pytesseract && tesseract --list-langs | grep ara\n"
            "  ocrmypdf -l ara --force-ocr data/hans_wehr.pdf data/hans_wehr_ocr.pdf"
        )
        raise typer.Exit(2)


def _import_vision():
    """Lazy import of pyobjc Vision framework with a helpful install hint."""
    try:
        import Vision
        from Foundation import NSURL
        return Vision, NSURL
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] pyobjc Vision framework not installed.\n"
            "Run:  uv pip install hans-wehr-mcp[pipeline-local]"
        )
        raise typer.Exit(2)


# ---------------------------------------------------------------------------
# Core OCR function
# ---------------------------------------------------------------------------

# Render DPI for Vision input — higher = better OCR, slower rendering.
# 200 DPI is a good balance for dictionary-quality Arabic text.
OCR_RENDER_DPI = 200

# Vision coordinate system has (0,0) at bottom-left; PyMuPDF uses top-left.
# All y-coordinates are flipped when converting.

# Two observations are considered the same "line" if their y-centres are
# within this many normalised units of each other (~5pt on an A5 page).
LINE_MERGE_THRESHOLD = 0.012

# Gap larger than this between lines → start a new block.
BLOCK_GAP_THRESHOLD = 0.04


def _flip_y(y_norm: float) -> float:
    """Vision uses bottom-left origin; convert to top-left (0 = top)."""
    return 1.0 - y_norm


def ocr_page(page: fitz.Page, page_number: int) -> dict:
    """OCR one PDF page using the macOS Vision framework.

    Renders the page to PNG at OCR_RENDER_DPI, passes it to
    VNRecognizeTextRequest with Arabic language, converts the resulting
    VNRecognizedTextObservation list into the same JSON structure that
    extract.py produces.
    """
    Vision, NSURL = _import_vision()

    # 1. Render page to PNG bytes via PyMuPDF
    scale = OCR_RENDER_DPI / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    png_bytes = pix.tobytes("png")

    page_width_pt = page.rect.width
    page_height_pt = page.rect.height

    # 2. Write PNG to a temp file (most reliable way to pass to Vision)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    try:
        os.write(tmp_fd, png_bytes)
        os.close(tmp_fd)

        # 3. Build and run Vision request
        url = NSURL.fileURLWithPath_(tmp_path)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLanguages_(["ar-SA", "ar"])
        request.setRecognitionLevel_(
            Vision.VNRequestTextRecognitionLevelAccurate
        )
        request.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        success, error = handler.performRequests_error_([request], None)

        if not success or error:
            log.warning("Page %d: Vision OCR failed — %s", page_number, error)
            return _empty_page(page_number, page_width_pt, page_height_pt)

        observations = request.results() or []

    finally:
        os.unlink(tmp_path)

    # 4. Convert VNRecognizedTextObservation list → our span/line/block format
    raw_spans = []
    for obs in observations:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        text = str(candidates[0].string()).strip()
        if not text:
            continue
        ocr_conf = float(candidates[0].confidence())

        # Vision bbox: normalised, bottom-left origin, y increases upward
        bbox = obs.boundingBox()
        x0_n = float(bbox.origin.x)
        y0_n = _flip_y(float(bbox.origin.y) + float(bbox.size.height))  # top in page coords
        x1_n = float(bbox.origin.x) + float(bbox.size.width)
        y1_n = _flip_y(float(bbox.origin.y))                             # bottom in page coords

        # Convert normalised → page points
        x0 = round(x0_n * page_width_pt, 2)
        y0 = round(y0_n * page_height_pt, 2)
        x1 = round(x1_n * page_width_pt, 2)
        y1 = round(y1_n * page_height_pt, 2)

        # Estimate font size from line height (Vision cap-height ≈ 65% of line box)
        line_height_pt = (y1_n - y0_n) * page_height_pt
        estimated_size = round(line_height_pt * 0.65, 2)

        raw_spans.append({
            "text": text,
            "y0_n": y0_n,  # normalised, for grouping
            "y_centre_n": (y0_n + y1_n) / 2,
            "span": {
                "text": text,
                "font": "vision-ocr",
                "size": max(estimated_size, 1.0),
                "flags": 0,
                "is_bold": False,   # Vision doesn't report weight
                "is_italic": False,
                "color": 0,
                "bbox": [x0, y0, x1, y1],
                "ocr_confidence": round(ocr_conf, 4),
            },
        })

    # 5. Group spans into lines (close y-centres), lines into blocks (y-gaps)
    raw_spans.sort(key=lambda s: s["y_centre_n"])
    blocks = _group_into_blocks(raw_spans)

    return {
        "page_number": page_number,
        "width": round(page_width_pt, 2),
        "height": round(page_height_pt, 2),
        "source": "vision_ocr",
        "blocks": blocks,
    }


def _group_into_blocks(raw_spans: list[dict]) -> list[dict]:
    """Group sorted spans into lines then blocks based on y-proximity."""
    if not raw_spans:
        return []

    # --- Group into lines ---
    lines: list[list[dict]] = []
    current_line: list[dict] = [raw_spans[0]]

    for span in raw_spans[1:]:
        prev_centre = sum(s["y_centre_n"] for s in current_line) / len(current_line)
        if abs(span["y_centre_n"] - prev_centre) <= LINE_MERGE_THRESHOLD:
            current_line.append(span)
        else:
            lines.append(current_line)
            current_line = [span]
    lines.append(current_line)

    # --- Group lines into blocks ---
    def _line_bbox(line_spans: list[dict]) -> tuple[float, float, float, float]:
        bboxes = [s["span"]["bbox"] for s in line_spans]
        return (
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        )

    blocks: list[dict] = []
    current_block_lines: list[list[dict]] = [lines[0]]
    prev_y1_n = lines[0][-1]["y_centre_n"]

    for line_spans in lines[1:]:
        line_y_n = line_spans[0]["y_centre_n"]
        if line_y_n - prev_y1_n > BLOCK_GAP_THRESHOLD:
            blocks.append(_build_block(current_block_lines, len(blocks)))
            current_block_lines = []
        current_block_lines.append(line_spans)
        prev_y1_n = line_spans[-1]["y_centre_n"]
    blocks.append(_build_block(current_block_lines, len(blocks)))

    return blocks


def _build_block(block_lines: list[list[dict]], block_no: int) -> dict:
    lines_out = []
    all_bboxes = []
    for line_no, line_spans in enumerate(block_lines):
        span_objs = [s["span"] for s in line_spans]
        bboxes = [s["bbox"] for s in span_objs]
        line_bbox = [
            min(b[0] for b in bboxes), min(b[1] for b in bboxes),
            max(b[2] for b in bboxes), max(b[3] for b in bboxes),
        ]
        all_bboxes.extend(bboxes)
        lines_out.append({
            "line_no": line_no,
            "bbox": line_bbox,
            "spans": span_objs,
        })

    block_bbox = [
        min(b[0] for b in all_bboxes), min(b[1] for b in all_bboxes),
        max(b[2] for b in all_bboxes), max(b[3] for b in all_bboxes),
    ]
    return {"block_no": block_no, "bbox": block_bbox, "lines": lines_out}


def _empty_page(page_number: int, width: float, height: float) -> dict:
    return {
        "page_number": page_number,
        "width": round(width, 2),
        "height": round(height, 2),
        "source": "vision_ocr",
        "blocks": [],
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _page_output_path(out_dir: Path, page_number: int) -> Path:
    return out_dir / f"page_{page_number:04d}.json"


def run_ocr(
    pdf_path: Path,
    out_dir: Path,
    start_page: int,
    end_page: int | None,
    overwrite: bool,
) -> int:
    _require_macos()

    doc = fitz.open(str(pdf_path))
    total = doc.page_count
    first = max(0, start_page - 1)
    last = min(total - 1, (end_page or total) - 1)
    page_range = range(first, last + 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"OCR pages {start_page}–{end_page or total} (macOS Vision)",
            total=len(page_range),
        )
        for page_idx in page_range:
            page_number = page_idx + 1
            out_path = _page_output_path(out_dir, page_number)

            if out_path.exists() and not overwrite:
                progress.advance(task)
                continue

            try:
                page_data = ocr_page(doc[page_idx], page_number)
                if not page_data["blocks"]:
                    log.warning("Page %d: Vision returned no text", page_number)
                out_path.write_text(
                    json.dumps(page_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                written += 1
            except Exception as exc:
                log.error("Page %d: OCR failed — %s", page_number, exc)
                errors += 1

            progress.advance(task)

    doc.close()
    console.print(
        f"[green]Done.[/green] {written} pages OCR'd, {errors} errors. "
        f"Output: {out_dir}"
    )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    pdf: Path = typer.Option(
        Path("data/hans_wehr.pdf"),
        "--pdf", "-p",
        help="Path to the scanned Hans Wehr PDF.",
    ),
    out: Path = typer.Option(
        Path("data/raw"),
        "--out", "-o",
        help="Output directory (same as hans-extract uses).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Process only the first 20 pages to verify OCR quality.",
    ),
    start: int = typer.Option(1, "--start", help="First page (1-indexed)."),
    end: int | None = typer.Option(None, "--end", help="Last page (inclusive)."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-OCR already-processed pages."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """OCR a scanned Arabic PDF using macOS Vision (Apple Silicon, free, on-device).

    Run hans-probe first to confirm the PDF is actually scanned.
    If it has selectable text, use hans-extract instead — it's faster and
    preserves font metadata that improves parser accuracy.

    Output writes to the same data/raw/ directory as hans-extract, so
    hans-parse works unchanged after this step.
    """
    _require_macos()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not pdf.exists():
        console.print(f"[bold red]Error:[/bold red] PDF not found: {pdf}")
        raise typer.Exit(2)

    if dry_run:
        console.print("[yellow]Dry-run:[/yellow] OCR'ing first 20 pages only.")
        run_ocr(pdf, out, 1, 20, overwrite)
    else:
        run_ocr(pdf, out, start, end, overwrite)


if __name__ == "__main__":
    app()
