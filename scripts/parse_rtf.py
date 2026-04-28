"""
scripts/parse_rtf.py
--------------------
Parse Apple OCR RTF output into the pipeline page_NNNN.json format.

When macOS Live Text copies Arabic text from a scanned PDF, the clipboard
contains RTF with bold/italic/size attributes.  The Vision Python API used by
ocr_pages.py cannot expose these attributes — it always reports is_bold=False
and is_italic=False.  This script converts Apple OCR RTF files into the same
JSON schema that hans-parse expects, so that the bold-flag detection path is
used instead of the column-position heuristic, giving more accurate headword
detection and removing the 0.60 confidence cap.

How to produce RTF files from macOS
------------------------------------
Option A – Preview (one page at a time):
  1. Open the scanned PDF in Preview.
  2. Tools → Text Selection (or press Cmd+Shift+T for Live Text).
  3. Select all text on the page (Cmd+A).
  4. Edit → Copy (copies RTF to clipboard).
  5. Paste into TextEdit, then File → Save As → Rich Text Format.
     Name the file  page_NNNN.rtf  matching the page number.

Option B – Command-line via pbpaste (macOS clipboard tool):
  After step 4 above:
    pbpaste -Prefer rtf > data/rtf/page_NNNN.rtf

Place all per-page .rtf files in a directory (e.g. data/rtf/) named like
  page_0042.rtf, page_0043.rtf, …
then run this script.

Usage
-----
  # Batch mode (directory of page_NNNN.rtf files):
  uv run python scripts/parse_rtf.py --rtf-dir data/rtf/ --out data/raw/

  # Single file (must supply page number explicitly):
  uv run python scripts/parse_rtf.py --rtf data/page_0042.rtf --page 42 --out data/raw/

Pipeline position
-----------------
  hans-probe          ← check PDF type (SCANNED)
  hans-ocr            ← Vision OCR (alternative to this script)
  THIS SCRIPT         ← Apple RTF OCR → page JSON (better bold detection)
  hans-verify-ocr     ← optional Tesseract cross-check (skip if using this script)
  hans-parse          ← works unchanged on RTF-sourced JSON
  hans-refine         ← LLM refinement (fewer entries need review)
  hans-import         ← load into SQLite

Requirements: only Python stdlib + typer + rich (already in pipeline deps).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

app = typer.Typer(
    help="Parse Apple OCR RTF files into pipeline page JSON (bold/italic preserved).",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RTF destination groups that should be skipped entirely.
# These appear as  {\fonttbl ...}  or  {\* \xyz ...}.
# ---------------------------------------------------------------------------
_SKIP_DESTINATIONS = frozenset({
    "fonttbl", "colortbl", "stylesheet", "info", "expandedcolortbl",
    "listtable", "listoverridetable", "revtbl", "rsidtbl", "pgdsctbl",
    "latentstyles", "datastore", "themedata", "colorschememapping",
    "pgptbl", "xmlns", "fldinst", "fldrslt",
})

# Tokenises RTF into: group delimiters, control words, hex bytes, plain text.
# Per the RTF spec a control word is \[a-z]+ with an optional signed-decimal
# parameter.  A single trailing space (the delimiter) is consumed.
_TOKEN_RE = re.compile(
    r"[{}]"                          # { or }
    r"|\\([a-z]+)(-?\d+)? ?"        # \word or \word-N (trailing space consumed)
    r"|\\'([\da-fA-F]{2})"          # \'xx  hex byte
    r"|\\\n"                         # escaped newline — line continuation, ignored
    r"|\\(.)"                        # control symbol  \\  \{  \}  \*  \-  \~  etc.
    r"|([^\\\{\}\r\n]+)"            # plain text (no backslash, no braces, no newlines)
    r"|[\r\n]",                      # bare newline — ignored in body text
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal parser state
# ---------------------------------------------------------------------------

@dataclass
class _Frame:
    """Per-group formatting state (inherits from parent on group open)."""
    bold: bool = False
    italic: bool = False
    size: float = 12.0        # points (RTF stores in half-points, we halve on read)
    skip: bool = False        # True when inside a skippable destination group
    optional: bool = False    # True when \* was just seen in this group
    first_word: bool = True   # True until the first control word in this group


# ---------------------------------------------------------------------------
# RTF parser
# ---------------------------------------------------------------------------

def parse_rtf(rtf_bytes: bytes) -> list[list[dict]]:
    """Parse RTF bytes into a list of paragraphs.

    Returns a list of paragraphs; each paragraph is a list of span dicts::

        {"text": str, "is_bold": bool, "is_italic": bool, "size": float}

    Paragraphs are delimited by \\par or \\line control words.
    Empty paragraphs (blank lines) are omitted.
    """
    # Decode bytes.  Apple generates UTF-8 or MacRoman/cp1252 RTF.
    content = _decode_rtf_bytes(rtf_bytes)

    # Pre-process Unicode escapes BEFORE tokenising so Arabic text ends up as
    # real Unicode in the token stream.  RTF encodes non-ASCII as \uN? where
    # N is a signed 16-bit decimal codepoint and ? is a replacement char.
    # Apple's RTF also uses \uN\'xx for the replacement.
    content = _expand_rtf_unicode(content)

    paragraphs: list[list[dict]] = []
    current_para: list[dict] = []
    current_buf: list[str] = []
    stack: list[_Frame] = [_Frame()]
    pending_optional = False

    def flush_buf() -> None:
        f = stack[-1]
        if current_buf and not f.skip:
            text = "".join(current_buf)
            if text:
                current_para.append({
                    "text": text,
                    "is_bold": f.bold,
                    "is_italic": f.italic,
                    "size": f.size,
                })
        current_buf.clear()

    def end_para() -> None:
        flush_buf()
        if current_para:
            paragraphs.append(list(current_para))
        current_para.clear()

    for m in _TOKEN_RE.finditer(content):
        tok = m.group(0)
        ctrl_word = m.group(1)
        ctrl_param = m.group(2)
        hex_byte = m.group(3)
        ctrl_sym = m.group(4)
        plain = m.group(5)

        # ── Group open ────────────────────────────────────────────────────
        if tok == "{":
            flush_buf()
            parent = stack[-1]
            frame = _Frame(
                bold=parent.bold,
                italic=parent.italic,
                size=parent.size,
                skip=parent.skip,
            )
            if pending_optional:
                frame.optional = True
                pending_optional = False
            stack.append(frame)

        # ── Group close ───────────────────────────────────────────────────
        elif tok == "}":
            flush_buf()
            if len(stack) > 1:
                stack.pop()

        # ── Control word ──────────────────────────────────────────────────
        elif ctrl_word:
            f = stack[-1]
            word = ctrl_word.lower()
            param = int(ctrl_param) if ctrl_param is not None else None

            # Skip known destination groups.  Two cases:
            #   {\fonttbl ...}      — named destination, first word in group
            #   {\* \unknown ...}   — optional destination, preceded by \*
            if (f.optional or f.first_word) and word in _SKIP_DESTINATIONS:
                f.skip = True
            f.optional = False
            f.first_word = False

            if f.skip:
                continue

            if word == "b":
                flush_buf()
                f.bold = (param != 0) if param is not None else True
            elif word == "i":
                flush_buf()
                f.italic = (param != 0) if param is not None else True
            elif word == "fs":
                if param is not None:
                    flush_buf()
                    f.size = param / 2.0    # half-points → points
            elif word == "plain":
                # Reset ALL character formatting to defaults.
                flush_buf()
                f.bold = False
                f.italic = False
                f.size = 12.0
            elif word == "par":
                end_para()
            elif word == "line":
                # Soft line break — treat as paragraph boundary.
                end_para()
            # All other control words (pard, f, cf, rtlch, ltrch, …) are
            # ignored — they don't affect the content we care about.

        # ── Hex byte ──────────────────────────────────────────────────────
        elif hex_byte and not stack[-1].skip:
            # Windows-1252 byte — typically Latin-1 extended or punctuation.
            try:
                char = bytes([int(hex_byte, 16)]).decode("cp1252", errors="replace")
                current_buf.append(char)
            except ValueError:
                pass

        # ── Control symbol ────────────────────────────────────────────────
        elif ctrl_sym:
            if ctrl_sym == "*":
                # Marks the NEXT group as an optional destination to skip.
                pending_optional = True
            elif not stack[-1].skip:
                if ctrl_sym in "\\{}":
                    current_buf.append(ctrl_sym)
                elif ctrl_sym == "~":
                    current_buf.append(" ")   # non-breaking space
                # \-  (soft hyphen) and others are silently dropped.

        # ── Plain text ────────────────────────────────────────────────────
        elif plain and not stack[-1].skip:
            current_buf.append(plain)

    end_para()
    return paragraphs


def _decode_rtf_bytes(data: bytes) -> str:
    """Try common encodings used by Apple's RTF output."""
    for enc in ("utf-8", "macroman", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _expand_rtf_unicode(content: str) -> str:
    """Replace \\uN (with optional replacement char) with the real Unicode char.

    RTF encodes non-ASCII characters as::

        \\uN?          N is a signed 16-bit decimal codepoint; ? is the ASCII replacement
        \\uN\\'xx     Same with a hex-byte replacement

    We convert both forms to the actual Unicode character so that Arabic text
    in the token stream is represented as-is rather than as control sequences.

    The optional replacement match group (``(?:...)?``) handles generators that
    set ``\\uc0`` (no replacement chars).
    """
    return re.sub(
        r"\\u(-?\d+)(?:\\'[\da-fA-F]{2}|\?)?",
        lambda m: chr(int(m.group(1)) % 65536),
        content,
    )


# ---------------------------------------------------------------------------
# Convert parsed paragraphs → pipeline page JSON
# ---------------------------------------------------------------------------

def _is_arabic_span(text: str) -> bool:
    """True if more than 30% of letter characters are Arabic script."""
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    letters = sum(1 for c in text if c.isalpha())
    return letters > 0 and arabic / letters > 0.30


def paragraphs_to_page_json(
    paragraphs: list[list[dict]],
    page_number: int,
) -> dict:
    """Convert RTF paragraphs to the pipeline page_NNNN.json schema.

    Key differences from Vision OCR JSON:
    - ``font`` is ``"rtf-ocr"`` (not ``"vision-ocr"``), so ``RawSpan.is_vision``
      is False and the parser uses bold-flag headword detection instead of
      column-position heuristics.
    - ``is_bold`` and ``is_italic`` are set from RTF formatting.
    - ``bbox`` is zeroed — RTF carries no spatial information.
    - Confidence is NOT capped at 0.60 (that cap is only for Vision OCR).
    """
    lines_out: list[dict] = []
    for line_no, para in enumerate(paragraphs):
        spans_out: list[dict] = []
        for raw in para:
            text = raw["text"].strip()
            if not text:
                continue
            is_bold = raw["is_bold"]
            is_italic = raw["is_italic"]
            spans_out.append({
                "text": text,
                "font": "rtf-ocr",
                "size": round(raw["size"], 2),
                "flags": (16 if is_bold else 0) | (2 if is_italic else 0),
                "is_bold": is_bold,
                "is_italic": is_italic,
                "is_arabic": _is_arabic_span(text),
                "column": "",          # no spatial column info in RTF
                "color": 0,
                "bbox": [0.0, 0.0, 0.0, 0.0],
            })
        if spans_out:
            lines_out.append({
                "line_no": line_no,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "spans": spans_out,
            })

    if lines_out:
        # Renumber lines sequentially within the single block.
        for i, ln in enumerate(lines_out):
            ln["line_no"] = i
        blocks: list[dict] = [{
            "block_no": 0,
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "lines": lines_out,
        }]
    else:
        blocks = []

    return {
        "page_number": page_number,
        "width": 0.0,
        "height": 0.0,
        "source": "rtf_ocr",
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _page_output_path(out_dir: Path, page_number: int) -> Path:
    return out_dir / f"page_{page_number:04d}.json"


def _infer_page_number(stem: str) -> int | None:
    """Extract page number from filenames like page_0042, page-42, or 42."""
    m = re.match(r"(?:page[_\-]?)?(\d+)$", stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    rtf: Path | None = typer.Option(
        None,
        "--rtf",
        help="Single RTF file.  Requires --page.",
        exists=False,
    ),
    rtf_dir: Path | None = typer.Option(
        None,
        "--rtf-dir",
        help="Directory of *.rtf files named page_NNNN.rtf.",
    ),
    page: int | None = typer.Option(
        None,
        "--page", "-p",
        help="Page number for --rtf mode.",
    ),
    out: Path = typer.Option(
        Path("data/raw"),
        "--out", "-o",
        help="Output directory (same as hans-extract / hans-ocr use).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-parse pages that already have a JSON output file.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Parse Apple OCR RTF files into pipeline page_NNNN.json.

    Two modes:

    \b
      --rtf FILE --page N   Parse one RTF file for page N.
      --rtf-dir DIR         Parse all page_NNNN.rtf files in DIR.

    \b
    Output goes to --out (default: data/raw/) in the same page_NNNN.json
    format that hans-parse, hans-verify-ocr and the rest of the pipeline use.

    \b
    Tip: name your RTF files page_0042.rtf so the page number is inferred
    automatically in batch mode.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if rtf is None and rtf_dir is None:
        console.print(
            "[bold red]Error:[/bold red] Provide either --rtf FILE or --rtf-dir DIR."
        )
        raise typer.Exit(2)
    if rtf is not None and rtf_dir is not None:
        console.print(
            "[bold red]Error:[/bold red] --rtf and --rtf-dir are mutually exclusive."
        )
        raise typer.Exit(2)

    out.mkdir(parents=True, exist_ok=True)

    # ── Single-file mode ──────────────────────────────────────────────────
    if rtf is not None:
        if not rtf.exists():
            console.print(f"[bold red]Error:[/bold red] File not found: {rtf}")
            raise typer.Exit(2)
        if page is None:
            # Try to infer from filename before failing.
            inferred = _infer_page_number(rtf.stem)
            if inferred is None:
                console.print(
                    "[bold red]Error:[/bold red] Cannot infer page number from "
                    f"filename {rtf.name!r}.  Use --page N."
                )
                raise typer.Exit(2)
            page = inferred

        out_path = _page_output_path(out, page)
        if out_path.exists() and not overwrite:
            console.print(
                f"[yellow]Skipping[/yellow] page {page} — {out_path.name} already exists "
                "(use --overwrite)."
            )
            return

        paragraphs = parse_rtf(rtf.read_bytes())
        page_json = paragraphs_to_page_json(paragraphs, page)
        out_path.write_text(
            json.dumps(page_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        n_spans = sum(len(ln["spans"]) for b in page_json["blocks"] for ln in b["lines"])
        console.print(
            f"[green]Written:[/green] {out_path.name}  "
            f"({len(paragraphs)} paragraphs, {n_spans} spans)"
        )
        return

    # ── Batch mode ────────────────────────────────────────────────────────
    assert rtf_dir is not None
    if not rtf_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Not a directory: {rtf_dir}")
        raise typer.Exit(2)

    rtf_files = sorted(rtf_dir.glob("*.rtf"))
    if not rtf_files:
        console.print(f"[bold red]Error:[/bold red] No *.rtf files found in {rtf_dir}")
        raise typer.Exit(2)

    written = errors = skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Parsing {len(rtf_files)} RTF files", total=len(rtf_files)
        )

        for rtf_file in rtf_files:
            page_number = _infer_page_number(rtf_file.stem)
            if page_number is None:
                console.print(
                    f"[yellow]Warning:[/yellow] Cannot infer page number from "
                    f"{rtf_file.name!r} — skipping."
                )
                errors += 1
                progress.advance(task)
                continue

            out_path = _page_output_path(out, page_number)
            if out_path.exists() and not overwrite:
                skipped += 1
                progress.advance(task)
                continue

            try:
                paragraphs = parse_rtf(rtf_file.read_bytes())
                page_json = paragraphs_to_page_json(paragraphs, page_number)
                out_path.write_text(
                    json.dumps(page_json, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                written += 1
                if verbose:
                    n_spans = sum(
                        len(ln["spans"])
                        for b in page_json["blocks"]
                        for ln in b["lines"]
                    )
                    log.debug("Page %d: %d paras, %d spans", page_number, len(paragraphs), n_spans)
            except Exception as exc:
                console.print(
                    f"[red]Error:[/red] Page {page_number} ({rtf_file.name}): {exc}"
                )
                errors += 1

            progress.advance(task)

    console.print(
        f"[green]Done.[/green]  Written: {written}  Skipped: {skipped}  Errors: {errors}.  "
        f"Output: {out}"
    )


if __name__ == "__main__":
    app()
