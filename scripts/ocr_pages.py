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
    """Lazy import of pyobjc Vision with a helpful install hint."""
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

# Hans Wehr has a two-column layout: Arabic (right) and English (left).
# This is the normalised x-coordinate of the centre gutter in Vision coordinates
# (bottom-left origin, 0.0–1.0).  Pages are ~595 pt wide; Arabic column starts
# around x=330 pt ≈ 0.55 normalised.  We process each column separately so
# Vision never has to handle mixed RTL Arabic + LTR English in the same region.
COLUMN_SPLIT_X = 0.52

# Two observations are considered the same "line" if their y-centres are
# within this many normalised units of each other (~5 pt on an A5 page).
LINE_MERGE_THRESHOLD = 0.012

# Gap larger than this between lines → start a new block.
BLOCK_GAP_THRESHOLD = 0.04


def _flip_y(y_norm: float) -> float:
    """Vision uses bottom-left origin; convert to top-left (0 = top)."""
    return 1.0 - y_norm


def _is_arabic_text(text: str) -> bool:
    """Return True if >30 % of the letter characters are Arabic script.

    Used to tag spans produced by the English-column pass that happen to
    contain inline Arabic examples, so the parser can handle them correctly.
    """
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    letter_chars = sum(1 for c in text if c.isalpha())
    return letter_chars > 0 and (arabic_chars / letter_chars) > 0.30


def _ocr_column_png(
    Vision,
    NSURL,
    png_bytes: bytes,
    languages: list[str],
    column_tag: str,
    x_offset_pt: float,
    clip_width_pt: float,
    page_height_pt: float,
    page_number: int,
) -> list[dict]:
    """Run Vision OCR on a pre-clipped column image.

    Because the PNG contains only one column, Vision never creates observations
    that span both columns — the RTL/LTR collision is physically impossible.

    Bounding boxes from Vision are normalised to the CLIP image (0–1 relative
    to clip width/height).  We convert them back to full page coordinates using
    x_offset_pt and clip_width_pt.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    try:
        os.write(tmp_fd, png_bytes)
        os.close(tmp_fd)

        url = NSURL.fileURLWithPath_(tmp_path)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLanguages_(languages)
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        success, error = handler.performRequests_error_([request], None)

        if not success or error:
            log.warning("Page %d [%s]: Vision failed — %s", page_number, column_tag, error)
            return []

        observations = request.results() or []
    finally:
        os.unlink(tmp_path)

    raw_spans = []
    for obs in observations:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        text = str(candidates[0].string()).strip()
        if not text:
            continue
        ocr_conf = float(candidates[0].confidence())

        # Vision bbox: normalised to the CLIP image, bottom-left origin.
        bbox = obs.boundingBox()
        x0_clip = float(bbox.origin.x)
        x1_clip = x0_clip + float(bbox.size.width)
        # Flip y from bottom-left to top-left
        y0_n = _flip_y(float(bbox.origin.y) + float(bbox.size.height))
        y1_n = _flip_y(float(bbox.origin.y))

        # Map clip-space x back to full page points:
        #   page_x = x_offset_pt + x_clip_normalised * clip_width_pt
        x0 = round(x_offset_pt + x0_clip * clip_width_pt, 2)
        x1 = round(x_offset_pt + x1_clip * clip_width_pt, 2)
        y0 = round(y0_n * page_height_pt, 2)
        y1 = round(y1_n * page_height_pt, 2)

        line_height_pt = (y1_n - y0_n) * page_height_pt
        estimated_size = round(line_height_pt * 0.65, 2)

        # is_arabic is content-based (Unicode), column is position-based (clip).
        is_arabic = _is_arabic_text(text)

        raw_spans.append({
            "y_centre_n": (y0_n + y1_n) / 2,
            "span": {
                "text": text,
                "font": "vision-ocr",
                "size": max(estimated_size, 1.0),
                "flags": 0,
                "is_bold": False,   # Vision does not report font weight
                "is_italic": False,
                "is_arabic": is_arabic,
                "column": column_tag,
                "color": 0,
                "bbox": [x0, y0, x1, y1],
                "ocr_confidence": round(ocr_conf, 4),
            },
        })

    return raw_spans


def ocr_page(page: fitz.Page, page_number: int) -> dict:
    """OCR one PDF page using the macOS Vision framework.

    Hans Wehr has a two-column layout (Arabic right, English left).
    Running a single Vision request on the full page causes RTL Arabic and
    LTR English to collide inside the same text observations.

    Fix: render each column as a SEPARATE PNG via PyMuPDF's clip parameter,
    then run Vision independently on each.  Vision can only see one script
    direction at a time so observations are always clean.

    Coordinate conversion:
      - Arabic (right) column clip: x ∈ [split_pt, page_width_pt]
        → Vision x_normalised maps back via:  page_x = split_pt + x_n * clip_w
      - English (left) column clip: x ∈ [0, split_pt]
        → Vision x_normalised maps back via:  page_x = x_n * split_pt
    """
    Vision, NSURL = _import_vision()

    scale = OCR_RENDER_DPI / 72.0
    mat = fitz.Matrix(scale, scale)

    page_width_pt = page.rect.width
    page_height_pt = page.rect.height
    split_pt = page_width_pt * COLUMN_SPLIT_X

    # Render each column to its own PNG — physically prevents cross-column bleed
    clip_arabic = fitz.Rect(split_pt, 0, page_width_pt, page_height_pt)
    clip_english = fitz.Rect(0, 0, split_pt, page_height_pt)

    png_arabic = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, clip=clip_arabic).tobytes("png")
    png_english = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, clip=clip_english).tobytes("png")

    common = dict(Vision=Vision, NSURL=NSURL, page_height_pt=page_height_pt, page_number=page_number)

    arabic_spans = _ocr_column_png(
        **common,
        png_bytes=png_arabic,
        # Arabic primary but include English so Vision can read the Latin
        # transliterations that follow every Arabic headword on the same line.
        languages=["ar-SA", "ar", "en-US"],
        column_tag="arabic",
        x_offset_pt=split_pt,
        clip_width_pt=page_width_pt - split_pt,
    )
    english_spans = _ocr_column_png(
        **common,
        png_bytes=png_english,
        languages=["en-US"],
        column_tag="english",
        x_offset_pt=0.0,
        clip_width_pt=split_pt,
    )

    if not arabic_spans and not english_spans:
        log.warning("Page %d: Vision returned no text from either column", page_number)
        return _empty_page(page_number, page_width_pt, page_height_pt)

    # Merge and sort by y-position (top → bottom reading order)
    all_spans = arabic_spans + english_spans
    all_spans.sort(key=lambda s: s["y_centre_n"])

    return {
        "page_number": page_number,
        "width": round(page_width_pt, 2),
        "height": round(page_height_pt, 2),
        "source": "vision_ocr",
        "blocks": _group_into_blocks(all_spans),
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
