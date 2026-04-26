# hans-wehr-mcp

An MCP (Model Context Protocol) server that exposes the Hans Wehr *Dictionary of Modern Written Arabic* (4th edition) as queryable tools for LLM clients.

Once set up, any MCP-compatible client — Claude Desktop, Continue.dev, or a custom agent — can call `lookup_root("كتب")` and get back all entries under that root, complete with definitions, verb forms, plural patterns, and source page numbers.

**New here?** Follow the step-by-step walkthrough in [GUIDE.md](GUIDE.md).

See [SPEC.md](SPEC.md) for the full architecture, data model, and accuracy strategy.

---

## Requirements

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- The Hans Wehr 4th edition PDF (not included)

---

## On-device pipeline (macOS, free)

If you are on Apple Silicon (M1/M2/M3/M4), the entire extraction pipeline runs **for free** — no API key, no cloud services, no cost:

| Step | Tool | Cost |
|---|---|---|
| PDF text extraction | PyMuPDF (CPU) | Free |
| OCR (if PDF is scanned) | macOS Vision framework | Free (Neural Engine) |
| LLM entry refinement | Ollama + Qwen2.5:7b (Metal GPU) | Free |
| MCP server | Pure Python | Free |

### Quick start (macOS on-device)

```bash
# 1. Install project (pipeline-local extra: no Anthropic SDK, adds pyobjc Vision)
uv pip install -e ".[pipeline-local]"

# 2. Install Ollama and pull a model
brew install ollama
ollama pull qwen2.5:7b   # ~4 GB, runs on Metal GPU
ollama serve             # leave this running in another terminal

# 3. Place your PDF
cp /path/to/hans_wehr.pdf data/hans_wehr.pdf

# 4. Check whether PDF has selectable text or is scanned
uv run hans-probe --pdf data/hans_wehr.pdf
```

**If the probe reports SELECTABLE** — the PDF has embedded text (most copies):

```bash
uv run hans-extract                    # Stage 1: PDF → data/raw/page_NNNN.json
uv run hans-parse                      # Stage 2: JSON → data/processed/entries.jsonl
uv run hans-refine --local             # Stage 3: Ollama cleanup (free, on-device)
uv run hans-import                     # Stage 4: JSONL → SQLite
uv run hans-validate                   # Sanity check counts and xref resolution
uv run hans-mcp                        # Start MCP server
```

**If the probe reports SCANNED** — pages are rasterised images, no embedded text:

```bash
uv run hans-ocr --pdf data/hans_wehr.pdf      # Vision OCR → data/raw/
uv run hans-verify-ocr --pdf data/hans_wehr.pdf  # cross-check with Tesseract
uv run hans-parse
uv run hans-refine --local
uv run hans-import
uv run hans-validate
uv run hans-mcp
```

> **Tip:** Run any step with `--dry-run` to test on the first 20 pages before
> committing to the full ~900-page run.

---

## Cloud pipeline (any OS, Anthropic API)

If you are on Linux/Windows, or prefer the highest-quality LLM refinement:

```bash
# Install with cloud pipeline dependencies (includes Anthropic SDK)
uv pip install -e ".[pipeline]"

# Add your Anthropic API key
cp .env.example .env
# edit .env → ANTHROPIC_API_KEY=sk-ant-...
```

Then run the same stages, omitting `--local`:

```bash
uv run hans-extract
uv run hans-parse
uv run hans-refine --dry-run   # check cost first
uv run hans-refine             # confirm, then run
uv run hans-import
uv run hans-validate
uv run hans-mcp
```

---

## Setup details

### 1. Install dependencies

Choose the right extra for your situation:

| Scenario | Command |
|---|---|
| MCP server only (already have a DB) | `uv pip install -e .` |
| macOS on-device pipeline (recommended) | `uv pip install -e ".[pipeline-local]"` |
| Cloud pipeline (Anthropic API) | `uv pip install -e ".[pipeline]"` |
| Everything | `uv pip install -e ".[pipeline,pipeline-local]"` |

### 2. Place the PDF

Copy your Hans Wehr 4th edition PDF to:

```
data/hans_wehr.pdf
```

The file is gitignored and never committed.

### 3. Probe the PDF

```bash
uv run hans-probe --pdf data/hans_wehr.pdf
```

This samples pages and reports whether the PDF has selectable text (`SELECTABLE`), a mix (`MIXED`), or only rasterised images (`SCANNED`). It also prints a font inventory table that helps calibrate the parser thresholds.

### 4a. Extract text (SELECTABLE PDF)

**Dry run first** (pages 1–20 only):

```bash
uv run hans-extract --dry-run
```

Review `data/raw/page_0001.json` … `page_0020.json`. Each file contains raw spans with font metadata. If text looks garbled, use OCR instead (see 4b).

**Full extraction:**

```bash
uv run hans-extract
```

### 4b. OCR pages (SCANNED PDF — macOS primary path)

```bash
# Dry run: first 20 pages
uv run hans-ocr --pdf data/hans_wehr.pdf --dry-run

# Full OCR (Vision framework, Neural Engine, no cost)
uv run hans-ocr --pdf data/hans_wehr.pdf
```

OCR output uses the same `data/raw/page_NNNN.json` format as `hans-extract`. Each span includes an `ocr_confidence` field.

### 4c. Verify OCR quality (cross-check Vision vs Tesseract)

After Vision OCR, run the verification layer to catch pages where one engine made systematic errors:

```bash
# Requires tesseract with Arabic data:
brew install tesseract tesseract-lang

# Dry run — shows which pages would be checked
uv run hans-verify-ocr --pdf data/hans_wehr.pdf --dry-run

# Full verification pass
uv run hans-verify-ocr --pdf data/hans_wehr.pdf
```

This runs OCRmyPDF (Tesseract backend) on the same PDF, then compares each page's text using character trigram Jaccard similarity. For each page it writes two new fields into the JSON:

| Field | Description |
|---|---|
| `ocr_agreement` | 0.0–1.0 similarity between Vision and Tesseract text |
| `tesseract_text` | Raw Tesseract output for the page |

Pages with `ocr_agreement < 0.40` have all span `ocr_confidence` values reduced by 0.20. This automatically routes those entries through the LLM refinement pass (step 6), so bad OCR pages get extra correction without any manual intervention.

Sample report output:

```
Page   Agreement   Status
----   ---------   ------
 312      18.3%    PENALISED    ← heavy diacritics confused one engine
  47      31.2%    PENALISED
 180      88.6%    OK
 181      91.2%    OK

Pages verified: 860  |  Average agreement: 83.4%  |  Low-agreement pages: 14
```

For non-macOS systems, skip `hans-ocr` and use OCRmyPDF directly as both the OCR source and the only pass:

```bash
ocrmypdf -l ara --force-ocr data/hans_wehr.pdf data/hans_wehr_ocr.pdf
uv run hans-extract --pdf data/hans_wehr_ocr.pdf   # extract from the Tesseract-annotated PDF
```

### 5. Parse extracted text

```bash
uv run hans-parse
```

Output: `data/processed/entries.jsonl` — one JSON dict per entry.

Check how many entries need review:

```bash
grep '"needs_review": true' data/processed/entries.jsonl | wc -l
```

### 6. LLM refinement pass (optional but recommended)

This step sends low-confidence entries to an LLM to correct parsing errors.

**On-device (free, macOS):**

```bash
# Make sure Ollama is running:
ollama serve

# Dry run (just shows counts, no LLM calls):
uv run hans-refine --local --dry-run

# Full refinement with default model (qwen2.5:7b):
uv run hans-refine --local

# Larger model for better accuracy (requires more RAM):
uv run hans-refine --local --model qwen2.5:14b
```

**Cloud (Anthropic API):**

```bash
uv run hans-refine --dry-run   # shows cost estimate
uv run hans-refine             # prompts for confirmation, then runs
```

For a typical run (~2,000 low-confidence entries) expect **$1–$3** via Anthropic.

You can also attach page thumbnails for better context on difficult entries:

```bash
uv run hans-refine --with-images --pdf data/hans_wehr.pdf --dry-run   # check cost
uv run hans-refine --with-images --pdf data/hans_wehr.pdf
```

Output: `data/processed/entries_refined.jsonl`

### 7. Import into SQLite

```bash
uv run hans-import
```

This creates `data/hans_wehr.db`, inserts all entries, builds FTS5 indexes, and stores cross-references.

After import, resolve cross-references and validate:

```bash
uv run python scripts/resolve_xrefs.py
uv run hans-validate
```

You should see:

```
Root count:   PASS  13,050 (expected 12,350–13,650)
Entry count:  PASS  61,200 (expected 51,000–69,000)
XRef resolution: 92.3%
```

### 8. Start the MCP server

```bash
uv run hans-mcp
```

The server communicates over stdin/stdout using the MCP protocol. Leave this running while you configure your client.

---

## Claude Desktop configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hans-wehr": {
      "command": "uv",
      "args": ["run", "hans-mcp"],
      "cwd": "/absolute/path/to/hans-wehr-mcp",
      "env": {
        "HANS_WEHR_DB_PATH": "/absolute/path/to/hans-wehr-mcp/data/hans_wehr.db"
      }
    }
  }
}
```

Restart Claude Desktop. You should see "hans-wehr" listed under connected MCP servers.

Try asking Claude: *"Look up the Arabic root كتب in Hans Wehr"*.

---

## Available MCP tools

| Tool | Description |
|---|---|
| `lookup_root(root)` | All entries under a root. Accepts Arabic script or transliteration. |
| `search_arabic(query, limit?)` | FTS search across Arabic words. Strips diacritics before searching. |
| `search_english(query, limit?)` | FTS search across English definitions (Porter-stemmed). |
| `get_entry(entry_id)` | Full detail for one entry including cross-references and parse metadata. |
| `list_roots(letter)` | All roots beginning with a given letter. |

---

## Running tests

```bash
uv run pytest
```

For coverage:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

---

## Project structure

```
hans-wehr-mcp/
├── src/hans_wehr/
│   ├── pipeline/
│   │   ├── extract.py      # Stage 1: PDF → per-page JSON (PyMuPDF)
│   │   ├── parser.py       # Stage 2: JSON → structured entry dicts
│   │   └── llm_refine.py   # Stage 3: LLM cleanup (Anthropic or Ollama)
│   ├── db/
│   │   ├── schema.sql      # SQLite schema with FTS5 virtual tables
│   │   └── queries.py      # All DB read queries (parameterised, injection-safe)
│   ├── mcp/
│   │   └── server.py       # MCP server and tool definitions
│   └── validation/
│       ├── sample_check.py # Random sample vs PDF page screenshot
│       ├── root_count.py   # Count validation against expected ranges
│       └── report.py       # Aggregate accuracy report
├── scripts/
│   ├── import_db.py        # Bulk import JSONL → SQLite
│   ├── ocr_pages.py        # macOS Vision OCR for scanned PDFs
│   ├── verify_ocr.py       # OCRmyPDF verification layer (Vision vs Tesseract)
│   ├── probe_pdf.py        # Pre-flight PDF text detection
│   └── resolve_xrefs.py    # Post-import cross-reference resolution
├── tests/
│   ├── test_parser.py      # Parser unit tests (no PDF required)
│   ├── test_queries.py     # DB query tests (in-memory SQLite)
│   └── test_mcp_server.py  # MCP tool dispatch tests
├── data/
│   ├── raw/                # Per-page JSON (gitignored)
│   └── processed/          # Structured JSONL (gitignored)
├── SPEC.md                 # Full architecture and accuracy specification
├── pyproject.toml
└── README.md
```

---

## Transliteration scheme

Hans Wehr uses the ALA-LC Arabic transliteration. Special characters that appear in the database:

| Arabic | Transliteration |
|---|---|
| ء | ʾ |
| ع | ʿ |
| ح | ḥ |
| خ | ḫ |
| ص | ṣ |
| ض | ḍ |
| ط | ṭ |
| ظ | ẓ |
| Long vowels | ā, ī, ū |

When querying tools, you can use either Arabic script or transliteration.

---

## Troubleshooting

**"No text blocks found" warnings during extraction**

Run `hans-probe` first. If it reports `SCANNED`, use `hans-ocr` (macOS) followed by `hans-verify-ocr`, or OCRmyPDF directly (any OS) before running `hans-parse`.

**"[tesseract] lots of diacritics — possibly poor OCR"**

This is a Tesseract warning, not an error. It fires when Arabic text has heavy tashkeel (vowel marks). Tesseract's default `ara` model handles unvowelled Arabic well but is weaker on fully diacritised text like Hans Wehr headwords.

Improve it by upgrading to the `tessdata_best` neural model:

```bash
# 1. Find your tessdata directory
brew --prefix tesseract   # e.g. /opt/homebrew/opt/tesseract

# 2. Download the high-accuracy Arabic model
curl -L -o /opt/homebrew/share/tessdata/ara.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/ara.traineddata
```

Re-run `hans-verify-ocr` — agreement scores will improve. Remaining bad pages are automatically routed to `hans-refine --local` for cleanup.

**"page already has text! — rasterising text and running OCR anyway"**

Harmless. Occurs when the PDF already has a partial text layer (even all-image PDFs can carry some metadata text). `--force-ocr` tells OCRmyPDF to ignore it and OCR the rendered image regardless. No action needed.

**Low OCR agreement scores across many pages**

If `hans-verify-ocr` shows average agreement below ~60%, check:
- The PDF may have heavy tashkeel (diacritics) — both engines can struggle. This is expected; LLM refinement will clean up the worst entries.
- Upgrade to `tessdata_best` as shown above.
- Try `hans-verify-ocr --verbose` to see per-span debug output.

**Ollama refuses to start / model not found**

```bash
ollama serve                  # start the server
ollama list                   # see what's pulled
ollama pull qwen2.5:7b        # pull the default model
```

If RAM is tight (< 8 GB), try `qwen2.5:3b` instead:

```bash
ollama pull qwen2.5:3b
uv run hans-refine --local --model qwen2.5:3b
```

**Low root count after import**

The parser detects roots by font size and boldness. Run `hans-probe` to see the font inventory, then adjust `ROOT_FONT_SIZE_MIN` and `DERIVED_FONT_SIZE_MIN` in `src/hans_wehr/pipeline/parser.py` to match your PDF's fonts.

**FTS search returns no results**

FTS5 indexes are built during import. If you inserted rows directly, re-run `hans-import` to rebuild them.

**MCP server not appearing in Claude Desktop**

Check that `cwd` and `HANS_WEHR_DB_PATH` in `claude_desktop_config.json` are absolute paths. Relative paths are not resolved correctly by the MCP host.
