"""
src/pipeline/llm_refine.py
--------------------------
Stage 3 (optional): send low-confidence parsed entries to an LLM and ask it
to return corrected structured JSON.

Two backends are supported:

  --local   Use a locally-running Ollama server (default model: qwen2.5:7b).
            Free, on-device, no API key required.  Recommended for Apple Silicon.
            Install: brew install ollama && ollama pull qwen2.5:7b

  (default) Use the Anthropic API (claude-sonnet-4-20250514).
            Requires ANTHROPIC_API_KEY in the environment.

This stage is NOT the primary parser — it is a cleanup pass. The structural
parse output and raw_text are always preserved in parse_metadata so the audit
trail is never lost.

Design:
  - Reads data/processed/entries.jsonl
  - Filters entries where needs_review == true
  - Batches them into LLM requests (up to BATCH_SIZE entries per call)
  - Before running (Anthropic only), prints a cost estimate and asks for confirmation
  - Writes refined entries to data/processed/entries_refined.jsonl
  - Entries that were NOT refined are copied through unchanged

Usage:
  # On-device (free, Apple Silicon recommended):
  uv run hans-refine --local
  uv run hans-refine --local --model qwen2.5:14b

  # Cloud (Anthropic):
  uv run hans-refine
  uv run hans-refine --dry-run  # cost estimate only, no API calls
  uv run hans-refine --with-images --pdf data/hans_wehr.pdf
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

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

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

BATCH_SIZE = 10           # entries per LLM call
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

    Returns a list of Anthropic content blocks (may include image blocks).
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


def _build_text_prompt(batch: list[dict]) -> str:
    """Build a plain-text prompt for Ollama (no image blocks)."""
    lines = [
        "Correct each of the following dictionary entries. Return a JSON array — one object per entry.",
        f"Each object must match this schema:\n{ENTRY_SCHEMA}",
        "Entries to correct:\n",
    ]
    for i, entry in enumerate(batch):
        lines.append(f"--- Entry {i + 1} ---")
        lines.append(f"raw_text: {entry.get('raw_text', '')}")
        lines.append(f"current_arabic: {entry.get('arabic', '')}")
        lines.append(f"current_definition: {entry.get('definition', '')}")
        lines.append(f"warnings: {entry.get('warnings', [])}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anthropic API call with retry / rate-limit handling
# ---------------------------------------------------------------------------

def _call_anthropic_with_retry(
    client,  # anthropic.Anthropic
    user_content: list[dict],
) -> tuple[str, int, int]:
    """Call the Anthropic API, retrying on rate-limit errors.

    Returns (response_text, input_tokens, output_tokens).
    """
    import anthropic as _anthropic

    delay = INITIAL_RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            text = response.content[0].text if response.content else ""
            usage = response.usage
            return text, usage.input_tokens, usage.output_tokens

        except _anthropic.RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            log.warning("Rate limit hit, retrying in %.1f s (attempt %d/%d)", delay, attempt, MAX_RETRIES)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

        except _anthropic.APIStatusError as exc:
            if attempt == MAX_RETRIES or exc.status_code < 500:
                raise
            log.warning("API error %d, retrying in %.1f s", exc.status_code, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

    raise RuntimeError("Exhausted retries")  # unreachable, but satisfies type checker


# ---------------------------------------------------------------------------
# Ollama API call (stdlib urllib — no extra dependencies)
# ---------------------------------------------------------------------------

def _call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
) -> str:
    """Send a chat request to a local Ollama server.

    Uses only stdlib urllib so no extra dependency is needed.
    Returns the assistant message text.
    Raises urllib.error.URLError if Ollama is not reachable.
    """
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,  # low temperature for structured output
        },
    }).encode("utf-8")

    url = ollama_url.rstrip("/") + "/api/chat"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["message"]["content"]
    except urllib.error.URLError as exc:
        raise urllib.error.URLError(
            f"Cannot reach Ollama at {url}. "
            f"Is it running? Start with: ollama serve\n"
            f"Original error: {exc.reason}"
        ) from exc


def _check_ollama_reachable(ollama_url: str, model: str) -> None:
    """Verify Ollama is running and the requested model is available."""
    tags_url = ollama_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] Cannot reach Ollama at {ollama_url}.\n"
            "Start it with:  [bold]ollama serve[/bold]\n"
            f"Details: {exc}"
        )
        raise typer.Exit(2)

    available = [m["name"] for m in data.get("models", [])]
    # Check for exact or prefix match (e.g. "qwen2.5:7b" matches "qwen2.5:7b-instruct-q4_K_M")
    if not any(m.startswith(model.split(":")[0]) for m in available):
        console.print(
            f"[bold red]Error:[/bold red] Model [cyan]{model}[/cyan] not found in Ollama.\n"
            f"Available models: {', '.join(available) or '(none)'}\n"
            f"Pull it with:  [bold]ollama pull {model}[/bold]"
        )
        raise typer.Exit(2)

    console.print(f"[green]Ollama reachable.[/green] Using model [cyan]{model}[/cyan].")


# ---------------------------------------------------------------------------
# Response parsing (shared between both backends)
# ---------------------------------------------------------------------------

def _parse_llm_response(raw_response: str, batch: list[dict], parse_method: str) -> list[dict]:
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
                updated = {**entry, **correction}
                updated["parse_method"] = parse_method
                updated["llm_model"] = parse_method  # record which backend was used
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
# Cost estimation (Anthropic only)
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


def refine_entries(
    input_path: Path,
    out_path: Path,
    dry_run: bool,
    threshold: float,
    limit: int | None,
    use_local: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    with_images: bool = False,
    pdf_path: Path | None = None,
) -> None:
    if with_images and use_local:
        console.print(
            "[yellow]Note:[/yellow] --with-images is not supported with --local (Ollama). "
            "Images will be ignored."
        )
        with_images = False

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

    if use_local:
        backend_label = f"Ollama ({ollama_model})"
        console.print(f"[cyan]Backend:[/cyan] {backend_label} — free, on-device, no cost.")
    else:
        cost = estimate_cost(len(to_refine), with_images=with_images)
        _print_cost_table(cost)

    if dry_run:
        console.print("[yellow]Dry-run mode:[/yellow] no LLM calls made.")
        return

    if len(to_refine) == 0:
        console.print("[green]No entries need refinement.[/green]")
        _write_output(out_path, pass_through)
        return

    if use_local:
        # Verify Ollama is available before starting the long batch run
        _check_ollama_reachable(ollama_url, ollama_model)
    else:
        if not typer.confirm(f"Proceed with ~${cost['approx_cost_usd']:.4f} in API costs?"):
            console.print("Aborted.")
            raise typer.Exit(0)

        import anthropic as _anthropic
        client = _anthropic.Anthropic()

    refined: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    parse_method = "llm_refined_local" if use_local else "llm_refined"

    batches = [to_refine[i:i + BATCH_SIZE] for i in range(0, len(to_refine), BATCH_SIZE)]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_label = f"Refining via {ollama_model}" if use_local else "Refining entries"
        task = progress.add_task(task_label, total=len(batches))

        for batch in batches:
            try:
                if use_local:
                    prompt = _build_text_prompt(batch)
                    response_text = _call_ollama(prompt, ollama_model, ollama_url)
                    corrected_batch = _parse_llm_response(response_text, batch, parse_method)
                    refined.extend(corrected_batch)
                else:
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

                        batch_image_tokens = len(page_images) * APPROX_IMAGE_TOKENS_PER_PAGE
                        log.debug(
                            "Batch: %d entries, %d unique page images (~%d image tokens)",
                            len(batch), len(page_images), batch_image_tokens,
                        )

                    user_content = _build_user_message(batch, page_images)
                    response_text, in_tok, out_tok = _call_anthropic_with_retry(client, user_content)
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok
                    corrected_batch = _parse_llm_response(response_text, batch, parse_method)
                    refined.extend(corrected_batch)

            except Exception as exc:
                log.error("Batch failed: %s — keeping originals", exc)
                for entry in batch:
                    entry.setdefault("warnings", []).append(f"llm_batch_error:{exc}")
                refined.extend(batch)

            progress.advance(task)

    if use_local:
        console.print(f"[green]Done.[/green] {len(refined)} entries refined via {ollama_model}.")
    else:
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
    local: bool = typer.Option(
        False,
        "--local",
        help=(
            "Use a local Ollama server instead of the Anthropic API. "
            "Free, on-device, no API key required. "
            "Requires Ollama running: brew install ollama && ollama serve"
        ),
    ),
    model: str = typer.Option(
        DEFAULT_OLLAMA_MODEL,
        "--model",
        help=f"Ollama model name (only used with --local). Default: {DEFAULT_OLLAMA_MODEL}",
    ),
    ollama_url: str = typer.Option(
        DEFAULT_OLLAMA_URL,
        "--ollama-url",
        help=f"Ollama server URL (only used with --local). Default: {DEFAULT_OLLAMA_URL}",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print cost estimate and exit without making any LLM calls.",
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
            "Requires --pdf. Run --dry-run first to see the cost breakdown. "
            "Not supported with --local."
        ),
    ),
    pdf: Path | None = typer.Option(
        None,
        "--pdf",
        help="Path to the source PDF. Required when --with-images is set.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Send low-confidence entries to an LLM for structured correction.

    Two backends:

    \b
    --local   Free, on-device via Ollama (recommended on Apple Silicon).
              brew install ollama && ollama pull qwen2.5:7b && ollama serve

    \b
    (default) Anthropic API (claude-sonnet-4-20250514). Requires ANTHROPIC_API_KEY.
              Estimates cost and asks for confirmation before making calls.
              Use --with-images for better context on garbled entries.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not input.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input}")
        raise typer.Exit(2)

    refine_entries(
        input, out, dry_run, threshold, limit,
        use_local=local,
        ollama_model=model,
        ollama_url=ollama_url,
        with_images=with_images,
        pdf_path=pdf,
    )


if __name__ == "__main__":
    app()
