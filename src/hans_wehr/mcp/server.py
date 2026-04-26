"""
src/mcp/server.py
-----------------
MCP server exposing the Hans Wehr dictionary as queryable tools.

Tools provided:
  lookup_root(root)           — all entries under a root (Arabic or transliteration)
  search_arabic(query, limit) — FTS search across Arabic words
  search_english(query, limit)— FTS search across English definitions
  get_entry(entry_id)         — full detail for one entry
  list_roots(letter)          — all roots beginning with a given letter

Usage (standalone):
  python -m src.mcp.server

Configure in Claude Desktop (~/.config/claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "hans-wehr": {
        "command": "python",
        "args": ["-m", "src.mcp.server"],
        "env": { "HANS_WEHR_DB_PATH": "/absolute/path/to/data/hans_wehr.db" }
      }
    }
  }
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from dotenv import load_dotenv
from mcp.server import Server

from hans_wehr.db.queries import (
    get_connection,
    get_entry_by_id,
    get_entries_for_root,
    get_root_by_arabic,
    get_root_by_transliteration,
    get_cross_references_for_entry,
    list_roots_by_letter,
    parse_plural_forms,
    search_arabic,
    search_english,
    search_transliteration,
    search_roots_by_transliteration,
    strip_diacritics,
)

load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "hans_wehr.db"


def _get_db_path() -> Path:
    env_val = os.environ.get("HANS_WEHR_DB_PATH")
    if env_val:
        return Path(env_val)
    return _DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

server = Server("hans-wehr")


def _db() -> sqlite3.Connection:
    """Open a fresh read-only connection for each request (thread-safe)."""
    return get_connection(str(_get_db_path()))


# ---------------------------------------------------------------------------
# Tool: list_tools
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="lookup_root",
            description=(
                "Return all dictionary entries under an Arabic root. "
                "Accepts Arabic script (voweled or unvoweled) or academic transliteration."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": (
                            "The Arabic root to look up. Examples: 'كتب', 'كَتَبَ', 'ktb', 'kataba'."
                        ),
                    }
                },
                "required": ["root"],
            },
        ),
        types.Tool(
            name="search_arabic",
            description=(
                "Full-text search across Arabic words in the dictionary. "
                "Diacritics are stripped before searching so voweled and unvoweled queries both work."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Arabic word or phrase to search for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 20, max 100).",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="search_english",
            description=(
                "Full-text search across English definitions in the dictionary. "
                "Uses Porter stemming so 'writing' matches 'write', 'wrote', etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "English word or phrase to search for in definitions.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 20, max 100).",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_entry",
            description=(
                "Return full details for a single dictionary entry by its integer ID. "
                "IDs are returned by other tools."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "integer",
                        "description": "The numeric entry ID.",
                    }
                },
                "required": ["entry_id"],
            },
        ),
        types.Tool(
            name="list_roots",
            description=(
                "List all dictionary roots beginning with a given letter. "
                "Accepts a single Arabic letter (e.g. 'ك') or a transliteration letter (e.g. 'k')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "A single Arabic letter or transliteration letter.",
                    }
                },
                "required": ["letter"],
            },
        ),
        types.Tool(
            name="search_transliteration",
            description=(
                "Search dictionary entries by their Hans Wehr transliteration. "
                "Accepts plain ASCII (e.g. 'kataba', 'husn', 'kitab') — diacritics like "
                "macrons and underdots are optional. "
                "Use this when the user types a word in Latin/English letters rather than Arabic script."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Transliteration to search for. Examples: 'kataba', 'kitab', "
                            "'husn', 'kātaba' (with or without diacritics)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 20, max 100).",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    """Convert a sqlite3.Row to a plain dict. Returns None if row is None."""
    if row is None:
        return None
    return dict(row)


def _entry_to_response(entry_row: sqlite3.Row, include_metadata: bool = False) -> dict:
    """Format a single entry row as a clean dict for API consumers."""
    d = dict(entry_row)

    # Deserialise plural_forms JSON column
    d["plural_forms"] = parse_plural_forms(d.get("plural_forms"))

    # Round confidence to 2 dp for readability
    if "confidence" in d:
        d["confidence"] = round(float(d["confidence"]), 2)

    if not include_metadata:
        # Strip heavy audit fields for list responses
        for key in ("raw_text", "extraction_warnings", "llm_model", "parse_method"):
            d.pop(key, None)

    return d


def _not_found_response(query_description: str) -> str:
    return json.dumps({"error": "not_found", "query": query_description}, ensure_ascii=False)


def _error_response(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: call_tool
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Dispatch incoming MCP tool calls to the appropriate handler."""

    db_path = _get_db_path()
    if not db_path.exists() and name != "list_roots":
        # Return a helpful error rather than crashing
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "error": "database_not_found",
                "message": (
                    f"Database not found at {db_path}. "
                    "Run the extraction pipeline first: see README.md."
                ),
            }),
        )]

    try:
        result = _dispatch(name, arguments)
    except Exception as exc:
        log.exception("Tool %s raised an exception", name)
        result = _error_response(f"Internal error: {exc}")

    return [types.TextContent(type="text", text=result)]


def _dispatch(name: str, arguments: dict) -> str:
    if name == "lookup_root":
        return _tool_lookup_root(arguments)
    if name == "search_arabic":
        return _tool_search_arabic(arguments)
    if name == "search_english":
        return _tool_search_english(arguments)
    if name == "search_transliteration":
        return _tool_search_transliteration(arguments)
    if name == "get_entry":
        return _tool_get_entry(arguments)
    if name == "list_roots":
        return _tool_list_roots(arguments)
    return _error_response(f"Unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_lookup_root(args: dict) -> str:
    raw_root = args.get("root", "").strip()
    if not raw_root:
        return _error_response("'root' argument is required")

    with _db() as conn:
        # 1. Try Arabic script lookup (strips diacritics internally)
        root_row = get_root_by_arabic(conn, raw_root)

        # 2. Try exact + ASCII-normalised transliteration lookup
        if root_row is None:
            root_row = get_root_by_transliteration(conn, raw_root)

        # 3. Bare Arabic consonants (strip diacritics and retry Arabic lookup)
        if root_row is None:
            bare = strip_diacritics(raw_root)
            if bare != raw_root:
                root_row = get_root_by_arabic(conn, bare)

        # 4. FTS fallback — handles partial matches and casual Latin spelling
        #    e.g. "katab" matching "kātaba", "ktb" matching Arabic unvoweled form
        if root_row is None:
            fts_hits = search_roots_by_transliteration(conn, raw_root, limit=1)
            if fts_hits:
                root_row = fts_hits[0]

        if root_row is None:
            return _not_found_response(raw_root)

        entries = get_entries_for_root(conn, root_row["id"])

        return json.dumps({
            "root": {
                "id": root_row["id"],
                "arabic": root_row["arabic"],
                "arabic_unvoweled": root_row["arabic_unvoweled"],
                "transliteration": root_row["transliteration"],
                "page": root_row["page_number"],
                "entry_count": root_row["entry_count"],
            },
            "entries": [_entry_to_response(e) for e in entries],
            "total": len(entries),
        }, ensure_ascii=False, indent=2)


def _tool_search_arabic(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return _error_response("'query' argument is required")

    limit = min(int(args.get("limit", 20)), 100)

    with _db() as conn:
        results = search_arabic(conn, query, limit)

    if not results:
        return json.dumps({"results": [], "total": 0, "query": query}, ensure_ascii=False)

    return json.dumps({
        "results": results,
        "total": len(results),
        "query": query,
        "note": "Results ordered by FTS5 relevance rank. Confidence scores reflect parse quality.",
    }, ensure_ascii=False, indent=2)


def _tool_search_english(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return _error_response("'query' argument is required")

    limit = min(int(args.get("limit", 20)), 100)

    with _db() as conn:
        results = search_english(conn, query, limit)

    if not results:
        return json.dumps({"results": [], "total": 0, "query": query}, ensure_ascii=False)

    return json.dumps({
        "results": results,
        "total": len(results),
        "query": query,
        "note": "English search uses Porter stemming. Snippets marked with <b>…</b>.",
    }, ensure_ascii=False, indent=2)


def _tool_get_entry(args: dict) -> str:
    try:
        entry_id = int(args["entry_id"])
    except (KeyError, ValueError, TypeError):
        return _error_response("'entry_id' must be a valid integer")

    with _db() as conn:
        entry_row = get_entry_by_id(conn, entry_id)
        if entry_row is None:
            return _not_found_response(str(entry_id))

        xrefs = get_cross_references_for_entry(conn, entry_id)
        entry_dict = _entry_to_response(entry_row, include_metadata=True)

        # Deserialise extraction_warnings
        if "extraction_warnings" in entry_dict:
            try:
                entry_dict["extraction_warnings"] = json.loads(entry_dict["extraction_warnings"] or "[]")
            except json.JSONDecodeError:
                entry_dict["extraction_warnings"] = []

        entry_dict["cross_references"] = [
            {
                "ref_type": r["ref_type"],
                "to_arabic_raw": r["to_arabic_raw"],
                "to_entry_id": r["to_entry_id"],
                "to_arabic": r["to_arabic"],
                "resolved": bool(r["resolved"]),
            }
            for r in xrefs
        ]

    return json.dumps(entry_dict, ensure_ascii=False, indent=2)


def _tool_search_transliteration(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return _error_response("'query' argument is required")

    limit = min(int(args.get("limit", 20)), 100)

    with _db() as conn:
        results = search_transliteration(conn, query, limit)

    if not results:
        return json.dumps(
            {"results": [], "total": 0, "query": query,
             "tip": "Try search_english for meaning-based lookup."},
            ensure_ascii=False,
        )

    return json.dumps({
        "results": results,
        "total": len(results),
        "query": query,
        "note": (
            "Matched against stored transliterations. "
            "Diacritics (macrons, underdots) are optional in the query."
        ),
    }, ensure_ascii=False, indent=2)


def _tool_list_roots(args: dict) -> str:
    letter = args.get("letter", "").strip()
    if not letter:
        return _error_response("'letter' argument is required")

    with _db() as conn:
        roots = list_roots_by_letter(conn, letter)

    if not roots:
        return json.dumps({"roots": [], "total": 0, "letter": letter}, ensure_ascii=False)

    return json.dumps({
        "roots": [
            {
                "id": r["id"],
                "arabic": r["arabic"],
                "arabic_unvoweled": r["arabic_unvoweled"],
                "transliteration": r["transliteration"],
                "page": r["page_number"],
                "entry_count": r["entry_count"],
            }
            for r in roots
        ],
        "total": len(roots),
        "letter": letter,
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    log.info("Starting Hans Wehr MCP server (DB: %s)", _get_db_path())
    mcp.server.stdio.run(server)


if __name__ == "__main__":
    main()
