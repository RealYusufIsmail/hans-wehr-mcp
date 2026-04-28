"""
src/pipeline/parser.py
----------------------
Stage 2 of the pipeline: parse per-page JSON (from extract.py) into
structured entry dicts.

Structural detection uses font size and boldness:
  - Root headword   : bold + large font (typically ≥ 11 pt in Hans Wehr)
  - Derived form    : bold + smaller font (typically 9–10.5 pt)
  - Definition text : regular weight, same or smaller size
  - POS tags        : italic spans within the definition line

Each parsed entry dict has the shape:

{
  "root_arabic":       "كَتَبَ",
  "root_unvoweled":    "كتب",
  "root_translit":     "kataba",
  "root_page":         42,
  "arabic":            "كِتَابٌ",
  "arabic_unvoweled":  "كتاب",
  "transliteration":   "kitāb",
  "part_of_speech":    "noun",
  "verb_form":         null,
  "plural_forms":      ["كُتُب"],
  "definition":        "book; letter…",
  "grammar_notes":     null,
  "page_number":       42,
  "confidence":        0.92,
  "needs_review":      false,
  "raw_text":          "<verbatim span text>",
  "warnings":          []
}

Usage:
  python -m src.pipeline.parser --raw data/raw/ --out data/processed/entries.jsonl
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

app = typer.Typer(help="Parse raw extracted JSON into structured dictionary entries.")
console = Console()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Font detection thresholds
# Adjust these after inspecting a sample of extracted pages.
# ---------------------------------------------------------------------------

# ── Selectable-PDF path (PyMuPDF / hans-extract) ──────────────────────────
# Font metadata (bold flag + size) is available.
ROOT_FONT_SIZE_MIN = 10.5    # pt — root headwords are in a larger font
DERIVED_FONT_SIZE_MIN = 8.5  # pt — derived forms in medium bold
SMALL_FONT_SIZE_MAX = 8.4    # pt — superscripts, footnotes (skip)

# ── Vision OCR path (hans-ocr / scanned PDF) ──────────────────────────────
# Vision doesn't report bold/italic.  We detect headwords by:
#   1. span.column == "arabic"  (right column of the page)
#   2. span contains any Arabic script character
#   3. estimated font size from bounding-box height
#
# Hans Wehr root headwords are ~11–13 pt; derived forms ~9–11 pt.
# Vision's size estimate (bbox_height × 0.65) tends to run small, so the
# thresholds here are lower than for selectable PDFs.
VISION_ROOT_SIZE_MIN = 9.0     # pt — root headwords in Vision output
VISION_DERIVED_SIZE_MIN = 7.5  # pt — derived forms in Vision output

# ---------------------------------------------------------------------------
# Arabic diacritics stripping
# ---------------------------------------------------------------------------
_DIACRITIC_RE = re.compile(r"[\u064B-\u065F\u0670]")


def strip_diacritics(text: str) -> str:
    return _DIACRITIC_RE.sub("", text)


# ---------------------------------------------------------------------------
# Transliteration extraction
# Transliteration appears right after the Arabic headword, typically in the
# same span or the next non-Arabic span, in a Latin script.
# ---------------------------------------------------------------------------
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
# Matches a run of Arabic characters (including spaces between them)
_ARABIC_RUN_RE = re.compile(r"[\u0600-\u06FF][\u0600-\u06FF\s]*[\u0600-\u06FF]|[\u0600-\u06FF]")
# Latin letters including composed characters used in Hans Wehr transliteration
_LATIN_WORD_RE = re.compile(r"[a-zA-Z\u00C0-\u024F\u02BC\u02BE][\w\u00C0-\u024F\u02BC\u02BE\-']*")


def is_arabic_text(text: str) -> bool:
    """True if the text contains Arabic characters."""
    return bool(_ARABIC_RE.search(text))


def contains_arabic(text: str) -> bool:
    """True if *any* Arabic character is present (lower threshold than is_arabic_text)."""
    return bool(_ARABIC_RE.search(text))


def extract_arabic_portion(text: str) -> str:
    """Return only the Arabic-script characters (and spaces between them) from text.

    Used with Vision OCR spans that contain mixed Arabic headword + Latin
    transliteration on the same line, e.g.:
        "بغضاء bigda and بغضة ,bugd بغض"  →  "بغضاء بغضة بغض"
    """
    runs = _ARABIC_RUN_RE.findall(text)
    return " ".join(r.strip() for r in runs if r.strip())


def extract_transliteration_from_span(text: str) -> str:
    """Extract the first Latin word (transliteration) from a mixed Vision span.

    Hans Wehr right-column lines look like:
        "[Arabic] translit optional-short-definition"
    The transliteration is the first Latin word (may include diacritics like ā, ḥ).
    """
    m = _LATIN_WORD_RE.search(text)
    if not m:
        return ""
    # Take everything up to the first comma / semicolon / opening paren
    candidate = re.split(r"[,;(]", text[m.start():])[0].strip()
    # Strip trailing punctuation and limit to ~30 chars (transliterations are short)
    candidate = candidate.rstrip(" .,:;")
    return candidate if len(candidate) <= 30 else ""


def is_latin_text(text: str) -> bool:
    """True if text is predominantly Latin (used to detect transliteration)."""
    if not text.strip():
        return False
    latin_chars = sum(1 for c in text if "LATIN" in unicodedata.name(c, "") or c.isascii())
    return latin_chars / max(len(text.strip()), 1) > 0.5


# ---------------------------------------------------------------------------
# Part-of-speech detection
# Hans Wehr uses abbreviated POS tags in italic.  Longer abbreviations must
# appear before shorter ones in the alternation so the regex matches greedily.
# ---------------------------------------------------------------------------
_POS_MAP: dict[str, str] = {
    "prop.n.": "proper_noun",
    "n.un.": "nomen_unitatis",
    "interj.": "particle",
    "prep.": "particle",
    "conj.": "particle",
    "pron.": "particle",
    "adj.": "adjective",
    "adv.": "adverb",
    "coll.": "collective_noun",
    "num.": "noun",
    "vn.": "verbal_noun",
    "ap.": "active_participle",
    "pp.": "passive_participle",
    "el.": "elative",
    "n.": "noun",
    "v.": "verb",
}

# Use negative-lookbehind/lookahead instead of \b because abbreviations end
# with "." which is a non-word char, so \b never matches after them.
_POS_RE = re.compile(
    r"(?<!\w)("
    + "|".join(re.escape(k) for k in sorted(_POS_MAP, key=len, reverse=True))
    + r")(?!\w)",
    re.IGNORECASE,
)

# Verb form pattern — Roman numerals I–X possibly with subforms (e.g. "II.", "IV")
_VERB_FORM_RE = re.compile(r"\b(X{0,1}(?:IX|IV|V?I{0,3}))\.?\b")

# Verb form section header: bold Roman numeral II–X at the start of a span.
_VERB_FORM_SECTION_RE = re.compile(r"^\s*(X{0,1}(?:IX|IV|V?I{0,3}))[\.\s]", re.IGNORECASE)

# Plural bracket pattern — e.g. "(pl. كُتُب)" or "(~ات)"
_PLURAL_RE = re.compile(r"\(pl\.\s*(.*?)\)", re.UNICODE)

# Cross-reference markers
_XREF_RE = re.compile(r"(?:see|cf\.?|→)\s+([\u0600-\u06FF\s]+)", re.UNICODE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RawSpan:
    text: str
    size: float
    is_bold: bool
    is_italic: bool
    font: str
    bbox: list[float]
    # Present only in Vision OCR pages (source == "vision_ocr")
    column: str = ""          # "arabic" | "english" | ""
    is_arabic: bool = False   # content-based: >30% Arabic chars
    ocr_confidence: float = 1.0

    @property
    def is_vision(self) -> bool:
        """True when this span was produced by Vision OCR (not PyMuPDF)."""
        return self.font == "vision-ocr"


@dataclass
class ParsedEntry:
    root_arabic: str
    root_unvoweled: str
    root_translit: str
    root_page: int
    arabic: str
    arabic_unvoweled: str
    transliteration: str
    part_of_speech: str | None
    verb_form: str | None
    plural_forms: list[str]
    definition: str
    grammar_notes: str | None
    page_number: int
    confidence: float
    needs_review: bool
    raw_text: str
    warnings: list[str] = field(default_factory=list)
    # Dictionary-structure fields
    entry_type: str | None = None           # root_verb | verb_form | verbal_noun | …
    parent_verb_form: str | None = None     # Roman numeral section this entry lives under
    definitions: list[str] = field(default_factory=list)  # definition split on semicolons

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(entry: ParsedEntry) -> float:
    """
    Start at 1.0 and subtract for each detected quality issue.
    The final score is clamped to [0.0, 1.0].
    """
    score = 1.0

    if not entry.transliteration:
        score -= 0.15
        if "no_transliteration" not in entry.warnings:
            entry.warnings.append("no_transliteration")

    if not entry.part_of_speech:
        score -= 0.10
        if "no_pos_tag" not in entry.warnings:
            entry.warnings.append("no_pos_tag")

    if not entry.arabic or not is_arabic_text(entry.arabic):
        score -= 0.20
        entry.warnings.append("no_arabic_text")

    if not entry.definition or len(entry.definition.strip()) < 3:
        score -= 0.20
        entry.warnings.append("empty_definition")

    # Combined penalty: both arabic and definition missing = completely unparsed
    if (not entry.arabic or not is_arabic_text(entry.arabic)) and (
        not entry.definition or len(entry.definition.strip()) < 3
    ):
        score -= 0.40
        entry.warnings.append("completely_unparsed")

    # Check for garbled Unicode (replacement characters)
    if "\uFFFD" in entry.raw_text:
        score -= 0.10
        entry.warnings.append("unicode_replacement_char")

    # Unrecognised unicode planes beyond Arabic, Latin, common punctuation
    for char in entry.arabic:
        cp = ord(char)
        if not (0x0000 <= cp <= 0x007F or 0x0600 <= cp <= 0x06FF or
                0x064B <= cp <= 0x065F or 0x0020 <= cp <= 0x002F):
            score -= 0.05
            entry.warnings.append("unexpected_unicode_in_arabic")
            break

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Core parsing logic
# ---------------------------------------------------------------------------

def _collect_spans(page_data: dict) -> list[RawSpan]:
    """Flatten all text spans from a page dict into an ordered list."""
    spans: list[RawSpan] = []
    for block in page_data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    spans.append(RawSpan(
                        text=span["text"],
                        size=span.get("size", 0.0),
                        is_bold=span.get("is_bold", False),
                        is_italic=span.get("is_italic", False),
                        font=span.get("font", ""),
                        bbox=span.get("bbox", [0, 0, 0, 0]),
                        column=span.get("column", ""),
                        is_arabic=span.get("is_arabic", False),
                        ocr_confidence=span.get("ocr_confidence", 1.0),
                    ))
    return spans


def _is_root_span(span: RawSpan) -> bool:
    """Heuristic: is this span an Arabic root headword?

    Selectable-PDF path: requires bold flag + large size.
    Vision OCR path: uses column position + any Arabic content + size.
      Root headwords appear in the RIGHT column and are the largest Arabic
      text on the line.  The font size estimate from Vision is the bbox height
      × 0.65, which runs smaller than the true pt size, so the threshold is
      lower than for selectable PDFs.
    """
    if span.is_vision:
        return (
            span.column == "arabic"
            and contains_arabic(span.text)
            and span.size >= VISION_ROOT_SIZE_MIN
        )
    return span.is_bold and span.size >= ROOT_FONT_SIZE_MIN and is_arabic_text(span.text)


def _is_derived_span(span: RawSpan) -> bool:
    """Heuristic: is this span a derived Arabic form (sub-entry under a root)?

    Same two-path logic as _is_root_span.
    """
    if span.is_vision:
        return (
            span.column == "arabic"
            and contains_arabic(span.text)
            and VISION_DERIVED_SIZE_MIN <= span.size < VISION_ROOT_SIZE_MIN
        )
    return (
        span.is_bold
        and DERIVED_FONT_SIZE_MIN <= span.size < ROOT_FONT_SIZE_MIN
        and is_arabic_text(span.text)
    )


def _extract_verb_form(text: str) -> str | None:
    """Extract Roman numeral verb form from a text fragment."""
    m = _VERB_FORM_RE.search(text)
    if m:
        form = m.group(1)
        # Sanity check: Hans Wehr uses I–X; reject noise matches
        try:
            val = _roman_to_int(form)
            if 1 <= val <= 10:
                return form
        except ValueError:
            pass
    return None


def _roman_to_int(s: str) -> int:
    """Convert Roman numeral string to integer. Raises ValueError if invalid."""
    vals = {"I": 1, "V": 5, "X": 10}
    result = 0
    prev = 0
    for ch in reversed(s.upper()):
        if ch not in vals:
            raise ValueError(f"Invalid Roman numeral character: {ch!r}")
        v = vals[ch]
        result += v if v >= prev else -v
        prev = v
    if result <= 0:
        raise ValueError("Non-positive Roman numeral")
    return result


def _is_verb_form_header(span: RawSpan) -> bool:
    """True if this bold Latin span is a Roman numeral verb form section header (II–X)."""
    if not span.is_bold or is_arabic_text(span.text):
        return False
    m = _VERB_FORM_SECTION_RE.match(span.text)
    if not m:
        return False
    try:
        val = _roman_to_int(m.group(1).upper())
        return 2 <= val <= 10
    except ValueError:
        return False


def expand_tilde(text: str, headword_unvoweled: str) -> str:
    """Replace every ~ with the current entry's unvoweled headword.

    Hans Wehr uses ~ as shorthand for the entry's root word in definitions
    and plural patterns (e.g. "~ات" = root + plural suffix ات).
    """
    if "~" not in text:
        return text
    return text.replace("~", headword_unvoweled)


def split_definitions(text: str) -> list[str]:
    """Split a definition string on semicolons, skipping those inside parentheses.

    Returns a list of stripped, non-empty definition fragments.
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ";" and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _infer_entry_type(
    pos: str | None,
    verb_form: str | None,
    parent_verb_form: str | None,
) -> str | None:
    """Map POS + verb form context to a semantic entry type label."""
    if pos == "verb":
        return "verb_form" if (verb_form and verb_form != "I") else "root_verb"
    _type_map: dict[str, str] = {
        "verbal_noun": "verbal_noun",
        "active_participle": "active_participle",
        "passive_participle": "passive_participle",
        "collective_noun": "collective_noun",
        "nomen_unitatis": "nomen_unitatis",
        "elative": "elative",
        "noun": "derived_noun",
        "proper_noun": "derived_noun",
        "adjective": "derived_adjective",
        "adverb": "derived_other",
        "particle": "derived_other",
    }
    return _type_map.get(pos or "")


def _extract_plurals(text: str) -> list[str]:
    """Extract plural forms from "(pl. X, Y)" patterns."""
    plurals: list[str] = []
    for m in _PLURAL_RE.finditer(text):
        raw = m.group(1)
        # Multiple plurals may be comma-separated
        for part in raw.split(","):
            p = part.strip()
            if p and is_arabic_text(p):
                plurals.append(strip_diacritics(p))  # store unvoweled
    return plurals


def _extract_pos(spans: list[RawSpan]) -> str | None:
    """Find POS tag in italic spans."""
    for span in spans:
        if span.is_italic:
            m = _POS_RE.search(span.text)
            if m:
                return _POS_MAP.get(m.group(1).lower())
    # Fall back to scanning definition text
    for span in spans:
        m = _POS_RE.search(span.text)
        if m:
            return _POS_MAP.get(m.group(1).lower())
    return None


def _extract_xrefs(text: str) -> list[dict]:
    """Find cross-reference markers in text, return list of xref dicts."""
    xrefs = []
    for m in _XREF_RE.finditer(text):
        target = m.group(1).strip()
        ref_type = "cf" if "cf" in m.group(0) else "see"
        xrefs.append({"to_arabic_raw": target, "ref_type": ref_type})
    return xrefs


def _spans_to_text(spans: list[RawSpan]) -> str:
    """Concatenate span texts with a single space."""
    return " ".join(s.text for s in spans).strip()


def parse_page(page_data: dict) -> Iterator[dict]:
    """
    Parse one page's span list into an iterator of entry dicts.

    State machine:
      SEEKING_ROOT → found a root span → READING_ENTRIES
      READING_ENTRIES → found another root → SEEKING_ROOT (emit accumulated entries)
                      → found derived span → emit previous entry, start new
                      → regular span → accumulate into current entry's definition
    """
    page_number: int = page_data["page_number"]
    spans = _collect_spans(page_data)

    current_root: dict | None = None
    current_entry_arabic: str = ""
    current_entry_arabic_unvoweled: str = ""
    current_entry_translit: str = ""
    current_entry_spans: list[RawSpan] = []
    current_verb_form_section: str | None = None  # tracks II–X section headers

    def _flush_entry() -> dict | None:
        """Build a ParsedEntry from current accumulation state and return its dict."""
        nonlocal current_entry_arabic, current_entry_arabic_unvoweled
        nonlocal current_entry_translit, current_entry_spans

        if not current_root or not current_entry_arabic:
            return None

        raw_text = _spans_to_text(current_entry_spans)
        definition_spans = [s for s in current_entry_spans if not (s.is_bold and is_arabic_text(s.text))]
        definition = _spans_to_text(definition_spans)

        # Clean definition: remove the arabic + transliteration preamble
        if current_entry_translit and definition.startswith(current_entry_translit):
            definition = definition[len(current_entry_translit):].lstrip(" ,;")

        # Expand tilde shorthand before further processing
        root_unvoweled = current_root.get("arabic_unvoweled", "")
        definition = expand_tilde(definition, root_unvoweled)

        pos = _extract_pos(current_entry_spans)
        verb_form = _extract_verb_form(raw_text)
        plurals = _extract_plurals(raw_text)
        xrefs = _extract_xrefs(raw_text)
        defs = split_definitions(definition.strip())
        entry_type = _infer_entry_type(pos, verb_form, current_verb_form_section)

        entry = ParsedEntry(
            root_arabic=current_root["arabic"],
            root_unvoweled=current_root["arabic_unvoweled"],
            root_translit=current_root["transliteration"],
            root_page=current_root["page_number"],
            arabic=current_entry_arabic,
            arabic_unvoweled=current_entry_arabic_unvoweled,
            transliteration=current_entry_translit,
            part_of_speech=pos,
            verb_form=verb_form,
            plural_forms=plurals,
            definition=definition.strip() or raw_text.strip(),
            grammar_notes=None,
            page_number=page_number,
            confidence=1.0,  # will be recomputed
            needs_review=False,
            raw_text=raw_text,
            warnings=[],
            entry_type=entry_type,
            parent_verb_form=current_verb_form_section,
            definitions=defs,
        )

        # Cross-references stored as a separate key (not in ParsedEntry dataclass)
        result = entry.to_dict()
        result["cross_references"] = xrefs

        # Score
        result["confidence"] = compute_confidence(entry)

        # Vision OCR pages lack bold detection and produce mixed-content spans.
        # Cap confidence so every Vision entry goes through LLM refinement.
        if any(s.is_vision for s in current_entry_spans):
            result["confidence"] = min(result["confidence"], 0.60)
            if "vision_ocr_source" not in result.get("warnings", []):
                result.setdefault("warnings", []).append("vision_ocr_source")
            result["parse_method"] = "vision_ocr"

        result["needs_review"] = result["confidence"] < 0.75

        # Reset accumulators
        current_entry_arabic = ""
        current_entry_arabic_unvoweled = ""
        current_entry_translit = ""
        current_entry_spans = []

        return result

    is_vision_page = page_data.get("source") == "vision_ocr"

    for span in spans:
        # Verb form section headers (bold "II.", "III. …" through "X.") reset the
        # section counter but don't start a new entry themselves.
        if _is_verb_form_header(span):
            m = _VERB_FORM_SECTION_RE.match(span.text)
            if m:
                current_verb_form_section = m.group(1).upper()
            continue

        if _is_root_span(span):
            e = _flush_entry()
            if e:
                yield e

            current_verb_form_section = None  # new root resets the section

            # For Vision OCR: the span contains "[Arabic headword] [transliteration] [definition]"
            # all mixed together.  Split out the Arabic and Latin parts.
            if span.is_vision:
                arabic = extract_arabic_portion(span.text) or span.text.strip()
                translit = extract_transliteration_from_span(span.text)
            else:
                arabic = span.text.strip()
                translit = ""

            current_root = {
                "arabic": arabic,
                "arabic_unvoweled": strip_diacritics(arabic),
                "transliteration": translit,
                "page_number": page_number,
            }
            current_entry_arabic = arabic
            current_entry_arabic_unvoweled = strip_diacritics(arabic)
            current_entry_translit = translit
            current_entry_spans = [span]

        elif _is_derived_span(span):
            e = _flush_entry()
            if e:
                yield e

            if span.is_vision:
                arabic = extract_arabic_portion(span.text) or span.text.strip()
                current_entry_translit = extract_transliteration_from_span(span.text)
            else:
                arabic = span.text.strip()
                current_entry_translit = ""

            current_entry_arabic = arabic
            current_entry_arabic_unvoweled = strip_diacritics(arabic)
            current_entry_spans = [span]

        else:
            if current_root is None:
                continue

            current_entry_spans.append(span)

            # Capture transliteration from the first Latin span after a headword.
            # For Vision OCR this was already extracted from the mixed span above;
            # for selectable PDFs we look at subsequent non-bold Latin spans.
            if (not current_entry_translit and is_latin_text(span.text)
                    and not span.is_bold and current_entry_arabic):
                candidate = re.split(r"[,;(]", span.text)[0].strip()
                if candidate and not is_arabic_text(candidate):
                    current_entry_translit = candidate
                    if current_root and not current_root["transliteration"]:
                        current_root["transliteration"] = candidate

    e = _flush_entry()
    if e:
        yield e


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    raw: Path = typer.Option(
        Path("data/raw"),
        "--raw",
        help="Directory containing per-page JSON files from extract.py.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    out: Path = typer.Option(
        Path("data/processed/entries.jsonl"),
        "--out", "-o",
        help="Output JSONL file (one entry dict per line).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Parse only the first 20 pages (same range as extract.py --dry-run).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Parse raw per-page JSON into structured dictionary entries (JSONL output)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    page_files = sorted(raw.glob("page_*.json"))
    if dry_run:
        page_files = page_files[:20]
        console.print("[yellow]Dry-run:[/yellow] parsing first 20 pages only.")

    if not page_files:
        console.print(f"[bold red]Error:[/bold red] No page_*.json files found in {raw}")
        raise typer.Exit(2)

    out.parent.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    low_confidence = 0

    with out.open("w", encoding="utf-8") as fh:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing pages", total=len(page_files))

            for page_file in page_files:
                try:
                    page_data = json.loads(page_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    log.error("Failed to load %s: %s", page_file, exc)
                    progress.advance(task)
                    continue

                for entry in parse_page(page_data):
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_entries += 1
                    if entry.get("needs_review"):
                        low_confidence += 1

                progress.advance(task)

    console.print(
        f"[green]Done.[/green] {total_entries} entries written to {out}. "
        f"{low_confidence} flagged needs_review."
    )


if __name__ == "__main__":
    app()
