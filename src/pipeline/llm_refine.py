"""
src/pipeline/llm_refine.py
--------------------------
Stage 3 (optional): send low-confidence parsed entries to the Anthropic API
and ask it to return corrected structured JSON.

This stage is NOT the primary parser — it is a cleanup pass. The structural
parse output and raw_text are always preserved in parse_metadata so the audit
trail is never lost.

Design:
  - Reads data/processed/entries.jsonl
  - Filters entries where needs_review == true
  - Batches them into API requests (up to BATCH_SIZE entries per call)
  - Before running, prints a cost estimate and asks for confirmation
  - Writes refined entries to data/processed/entries_refined.jsonl
  - Entries that were NOT refined are copied through unchanged

Usage:
  python -m src.pipeline.llm_refine --input data/processed/entries.jsonl \
      --out data/processed/entries_refined.jsonl
  python -m src.pipeline.llm_refine --dry-run  # cost estimate only, no API calls
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import anthropic
import fitz  # PyMuPDF — used for page thumbnail rendering
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

load_dotenv()

app = typer.Typer(help="LLM refinement pass for low-confidence parsed entries.")
console = Console()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
BATCH_SIZE = 10           # entries per API call
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2.0  # seconds

# Approximate costs (USD per 1M tokens) — update if Anthropic changes pricing
INPUT_COST_PER_MTOK = 3.00
OUTPUT_COST_PER_MTOK = 15.00

# Conservative token estimates per entry for cost estimation (text-only mode)
APPROX_INPUT_TOKENS_PER_ENTRY = 400
APPROX_OUTPUT_TOKENS_PER_ENTRY = 200

# Image rendering constants
# Keep thumbnails narrow: Anthropic charges per tile (32×32px grid).
# A 1200px-wide image at typical dictionary aspect ratio (~1500px tall) costs
# roughly ceil(1200/32) × ceil(1500/32) × 2 ≈ 2888 tokens.  We use 3000 as a
# safe upper bound.  Wider images multiply this quickly — do NOT raise this
# limit without re-running the cost estimate.
MAX_IMAGE_WIDTH_PX = 1200
APPROX_IMAGE_TOKENS_PER_PAGE = 3000  # vision token estimate per thumbnail

# Confidence threshold — entries below this are sent for refinement
CONFIDENCE_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Arabic lexicographer specialising in the Hans Wehr Dictionary of Modern Written Arabic.
You will be given raw text extracted from the dictionary PDF along with a partially-parsed entry.
Your job is to return a corrected JSON object with accurate fields.

Rules:
1. Return ONLY valid JSON — no prose, no markdown code fences.
2. Preserve the exact Arabic text as it appears in the raw_text — do not transliterate yourself.
3. If a field is genuinely absent from the raw_text, return null for that field.
4. The 'definition' field must be the full English definition as printed — do not summarise.
5. Valid part_of_speech values: verb | noun | adjective | adverb | particle | phrase | proper_noun
6. Valid verb_form values: I | II | III | IV | V | VI | VII | VIII | IX | X | null
7. plural_forms must be a JSON array of Arabic strings (may be empty []).
8. Do not invent information not present in raw_text.
"""

ENTRY_SCHEMA = """\
{
  "arabic": "<Arabic word in script>",
  "arabic_unvoweled": "<same without diacritics>",
  "transliteration": "<Hans Wehr academic transliteration or null>",
  "part_of_speech": "<verb|noun|adjective|adverb|particle|phrase|proper_noun|null>",
  "verb_form": "<I–X or null>",
  "plural_forms": ["<Arabic plural>"],
  "definition": "<full English definition>",
  "grammar_notes": "<grammar note string or null>",
  "confidence": <float 0.0–1.0 — your confidence in this correction>,
  "correction_notes": "<brief explanation of what you changed>"
}
"""


# ---------------------------------------------------------------------------
# Page thumbnail rendering
# ---------------------------------------------------------------------------

def _render_page_thumbnail(pdf_path: Path, page_number: int) -> str | None:
    """Render one PDF page as a base64-encoded PNG, capped at MAX_IMAGE_WIDTH_PX.

    Returns the base64 string, or None if rendering fails (e.g. page is a scan
    with no vector content, or the PDF cannot be opened).

    The scale factor is chosen so that the output width never exceeds
    MAX_IMAGE_WIDTH_PX.  For a typical A5 dictionary page (~419 pt wide at
    72 dpi) this works out to roughly 2.05× zoom.
    """
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number - 1]  # page_number is 1-indexed

        # Compute scale to cap width at MAX_IMAGE_WIDTH_PX
        page_width_pt = page.rect.width  # points at 72 dpi
        target_dpi = (MAX_IMAGE_WIDTH_PX / page_width_pt) * 72
        scale = target_dpi / 72
        mat = fitz.Matrix(scale, scale)

        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        png_bytes = pix.tobytes("png")
        doc.close()

        actual_width = pix.width
        if actual_width > MAX_IMAGE_WIDTH_PX:
            # Defensive: should not happen due to scale calculation, but guard anyway
            log.warning(
                "Page %d thumbnail is %dpx wide (limit %dpx) — skipping image",
                page_number, actual_width, MAX_IMAGE_WIDTH_PX,
            )
            return None

        return base64.standard_b64encode(png_bytes).decode("ascii")

    except Exception as exc:
        log.warning("Could not render thumbnail for page %d: %s", page_number, exc)
        return None


def _build_user_message(
    batch: list[dict],
    page_images: dict[int, str] | None = None,
) -> list[dict]:
    """Build the user message content blocks for one API call.

    Returns a list of Anthropic content blocks (text and optionally image).
    page_images maps page_number → base64 PNG string.
    """
    text_lines = [
        "Correct each of the following dictionary entries. Return a JSON array — one object per entry.\n",
        f"Each object must match this schema:\n{ENTRY_SCHEMA}\n",
        "Entries to correct:\n",
    ]
    for i, entry in enumerate(batch):
        text_lines.append(f"--- Entry {i + 1} ---")
        text_lines.append(f"raw_text: {entry.get('raw_text', '')}")
        text_lines.append(f"current_arabic: {entry.get('arabic', '')}")
        text_lines.append(f"current_definition: {entry.get('definition', '')}")
        text_lines.append(f"warnings: {entry.get('warnings', [])}")
        if page_images:
            page_num = entry.get("page_number")
            if page_num and page_num in page_images:
                text_lines.append(f"(page image attached below for entry {i + 1})")
        text_lines.append("")

    content: list[dict] = [{"type": "text", "text": "\n".join(text_lines)}]

    # Append unique page images (one per distinct page in the batch)
    if page_images:
        seen_pages: set[int] = set()
        for entry in batch:
            page_num = entry.get("page_number")
            if page_num and page_num in page_images and page_num not in seen_pages:
                seen_pages.add(page_num)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": page_images[page_num],
                    },
                })

    return content


# ---------------------------------------------------------------------------
# API call with retry / rate-limit handling
# ---------------------------------------------------------------------------

def _call_api_with_retry(
    client: anthropic.Anthropic,
    user_content: list[dict],
) -> tuple[str, int, int]:
    """Call the Anthropic API, retrying on rate-limit errors.

    user_content is a list of Anthropic content blocks (may include image blocks).
    Returns (response_text, input_tokens, output_tokens).
    """
    delay = INITIAL_RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            text = response.content[0].text if response.content else ""
            usage = response.usage
            return text, usage.input_tokens, usage.output_tokens

        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            log.warning("Rate limit hit, retrying in %.1f s (attempt %d/%d)", delay, attempt, MAX_RETRIES)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

        except anthropic.APIStatusError as exc:
            if attempt == MAX_RETRIES or exc.status_code < 500:
                raise
            log.warning("API error %d, retrying in %.1f s", exc.status_code, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

    raise RuntimeError("Exhausted retries")  # unreachable, but satisfies type checker


def _parse_llm_response(raw_response: str, batch: list[dict]) -> list[dict]:
    """Parse the JSON array from the LLM response.

    If parsing fails, return the original entries unchanged with a warning added.
    """
    # Strip markdown code fences if the model included them
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        corrections = json.loads(text)
    except json.JSONDecodeError as exc:
        log.error("LLM returned invalid JSON: %s — falling back to original entries", exc)
        for entry in batch:
            entry.setdefault("warnings", []).append("llm_invalid_json_response")
        return batch

    if not isinstance(corrections, list):
        corrections = [corrections]

    merged = []
    for i, entry in enumerate(batch):
        if i < len(corrections):
            correction = corrections[i]
            if isinstance(correction, dict):
                # Merge: LLM fields overwrite structural parse, but preserve audit fields
                updated = {**entry, **correction}
                updated["parse_method"] = "llm_refined"
                updated["llm_model"] = MODEL
                # Keep the original raw_text
                updated["raw_text"] = entry.get("raw_text", "")
                merged.append(updated)
            else:
                merged.append(entry)
        else:
            log.warning("LLM returned fewer corrections than batch size; keeping original for entry %d", i)
            merged.append(entry)

    return merged


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(n_entries: int, with_images: bool = False) -> dict:
    """Estimate API cost for refining *n_entries* entries.

    When *with_images* is True, adds vision token overhead per batch.
    Each batch may reference up to BATCH_SIZE distinct pages; in practice
    most entries in a batch share a page, so we conservatively assume
    1 unique page image per 3 entries.
    """
    n_batches = (n_entries + BATCH_SIZE - 1) // BATCH_SIZE
    text_input_tokens = n_entries * APPROX_INPUT_TOKENS_PER_ENTRY
    text_output_tokens = n_entries * APPROX_OUTPUT_TOKENS_PER_ENTRY

    # Image cost: ~1 unique page per 3 entries at 3000 tokens per thumbnail.
    # This is a conservative over-estimate — tightly clustered pages will cost less.
    image_input_tokens = 0
    if with_images:
        approx_unique_pages = max(1, n_entries // 3)
        image_input_tokens = approx_unique_pages * APPROX_IMAGE_TOKENS_PER_PAGE

    total_input_tokens = text_input_tokens + image_input_tokens
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_MTOK
    output_cost = (text_output_tokens / 1_000_000) * OUTPUT_COST_PER_MTOK

    return {
        "n_entries": n_entries,
        "n_batches": n_batches,
        "approx_text_input_tokens": text_input_tokens,
        "approx_image_input_tokens": image_input_tokens,
        "approx_input_tokens": total_input_tokens,
        "approx_output_tokens": text_output_tokens,
        "approx_cost_usd": round(input_cost + output_cost, 4),
        "with_images": with_images,
    }


def _print_cost_table(cost: dict) -> None:
    table = Table(title="Cost Estimate", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Entries to refine", str(cost["n_entries"]))
    table.add_row("API batches", str(cost["n_batches"]))
    table.add_row("Est. text input tokens", f"{cost['approx_text_input_tokens']:,}")
    if cost.get("with_images"):
        table.add_row(
            "Est. image input tokens",
            f"{cost['approx_image_input_tokens']:,}  "
            f"[dim](≈{cost['approx_image_input_tokens'] // max(cost['n_entries'] // 3, 1):,} tok/page, "
            f"max {MAX_IMAGE_WIDTH_PX}px wide)[/dim]",
        )
    table.add_row("Est. total input tokens", f"{cost['approx_input_tokens']:,}")
    table.add_row("Est. output tokens", f"{cost['approx_output_tokens']:,}")
    table.add_row("Est. total cost (USD)", f"${cost['approx_cost_usd']:.4f}")
    console.print(table)


# ---------------------------------------------------------------------------
# Main refinement logic
# ---------------------------------------------------------------------------

import re  # noqa: E402 (imported again for _parse_llm_response — already imported above)


def refine_entries(
    input_path: Path,
    out_path: Path,
    dry_run: bool,
    threshold: float,
    limit: int | None,
    with_images: bool = False,
    pdf_path: Path | None = None,
) -> None:
    if with_images and (pdf_path is None or not pdf_path.exists()):
        console.print(
            "[bold red]Error:[/bold red] --with-images requires --pdf pointing to the source PDF."
        )
        raise typer.Exit(2)

    # Load all entries
    all_entries: list[dict] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    all_entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed JSONL line: %s", exc)

    console.print(f"Loaded {len(all_entries)} total entries from {input_path}")

    # Split: needs refinement vs pass-through
    to_refine = [e for e in all_entries if e.get("needs_review") and e.get("confidence", 1.0) < threshold]
    pass_through = [e for e in all_entries if not (e.get("needs_review") and e.get("confidence", 1.0) < threshold)]

    if limit:
        to_refine = to_refine[:limit]

    console.print(f"{len(to_refine)} entries flagged for refinement, {len(pass_through)} pass-through.")

    cost = estimate_cost(len(to_refine), with_images=with_images)
    _print_cost_table(cost)

    if dry_run:
        console.print("[yellow]Dry-run mode:[/yellow] no API calls made.")
        return

    if len(to_refine) == 0:
        console.print("[green]No entries need refinement.[/green]")
        # Still write the pass-through entries
        _write_output(out_path, pass_through)
        return

    if not typer.confirm(f"Proceed with ~${cost['approx_cost_usd']:.4f} in API costs?"):
        console.print("Aborted.")
        raise typer.Exit(0)

    client = anthropic.Anthropic()

    refined: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0

    batches = [to_refine[i:i + BATCH_SIZE] for i in range(0, len(to_refine), BATCH_SIZE)]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Refining entries", total=len(batches))

        for batch in batches:
            # Render page thumbnails for this batch (de-duplicated by page number)
            page_images: dict[int, str] | None = None
            if with_images and pdf_path:
                page_images = {}
                for entry in batch:
                    page_num = entry.get("page_number")
                    if page_num and page_num not in page_images:
                        thumbnail = _render_page_thumbnail(pdf_path, page_num)
                        if thumbnail:
                            page_images[page_num] = thumbnail

                # Estimate image token cost for this specific batch and log it
                batch_image_tokens = len(page_images) * APPROX_IMAGE_TOKENS_PER_PAGE
                log.debug(
                    "Batch: %d entries, %d unique page images (~%d image tokens)",
                    len(batch), len(page_images), batch_image_tokens,
                )

            user_content = _build_user_message(batch, page_images)
            try:
                response_text, in_tok, out_tok = _call_api_with_retry(client, user_content)
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                corrected_batch = _parse_llm_response(response_text, batch)
                refined.extend(corrected_batch)
            except Exception as exc:
                log.error("Batch failed: %s — keeping originals", exc)
                for entry in batch:
                    entry.setdefault("warnings", []).append(f"llm_batch_error:{exc}")
                refined.extend(batch)
            progress.advance(task)

    actual_cost = (
        (total_input_tokens / 1_000_000) * INPUT_COST_PER_MTOK
        + (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_MTOK
    )
    console.print(
        f"[green]Done.[/green] Used {total_input_tokens:,} input + "
        f"{total_output_tokens:,} output tokens. "
        f"Actual cost: ~${actual_cost:.4f}"
    )

    # Merge refined + pass-through and write output
    # Preserve original page order by re-sorting on page_number then entry order
    all_output = pass_through + refined
    all_output.sort(key=lambda e: (e.get("page_number", 0), e.get("arabic", "")))
    _write_output(out_path, all_output)


def _write_output(out_path: Path, entries: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    console.print(f"Wrote {len(entries)} entries to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    input: Path = typer.Option(
        Path("data/processed/entries.jsonl"),
        "--input", "-i",
        help="JSONL file from parser.py.",
    ),
    out: Path = typer.Option(
        Path("data/processed/entries_refined.jsonl"),
        "--out", "-o",
        help="Output JSONL with refined entries merged in.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print cost estimate and exit without making any API calls.",
    ),
    threshold: float = typer.Option(
        CONFIDENCE_THRESHOLD,
        "--threshold",
        help="Confidence score below which entries are sent for refinement.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Maximum number of entries to refine (for testing).",
    ),
    with_images: bool = typer.Option(
        False,
        "--with-images",
        help=(
            "Attach a page thumbnail (PNG, max 1200px wide) to each API batch. "
            "Gives the model visual context but significantly increases token cost. "
            "Requires --pdf. Run --dry-run first to see the cost breakdown."
        ),
    ),
    pdf: Path | None = typer.Option(
        None,
        "--pdf",
        help="Path to the source PDF. Required when --with-images is set.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Send low-confidence entries to Claude for structured correction.

    Always estimates cost before making API calls and asks for confirmation.

    Use --with-images to attach page thumbnails for better context on entries
    with garbled text.  Run --dry-run first to see the full cost breakdown
    before committing to an expensive image-enabled pass.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not input.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input}")
        raise typer.Exit(2)

    refine_entries(input, out, dry_run, threshold, limit, with_images=with_images, pdf_path=pdf)


if __name__ == "__main__":
    app()
