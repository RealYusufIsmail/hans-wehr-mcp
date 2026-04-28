"""
tests/test_rtf_parser.py
------------------------
Unit tests for scripts/parse_rtf.py

Covers:
- Basic bold / italic / size extraction
- Group inheritance and reset (plain)
- Paragraph breaks (par, line)
- Hex byte decoding (hex escapes)
- Arabic Unicode via RTF decimal escapes
- Skippable destination groups (fonttbl, optional destinations)
- _is_arabic_span heuristic
- paragraphs_to_page_json output schema
- _infer_page_number filename helper
"""

from __future__ import annotations

import json

import pytest

from scripts.parse_rtf import (
    _infer_page_number,
    detect_language,
    paragraphs_to_page_json,
    parse_rtf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rtf(body: str) -> bytes:
    """Wrap body text in a minimal RTF envelope."""
    return (r"{\rtf1\ansi " + body + "}").encode()


# ---------------------------------------------------------------------------
# parse_rtf — basic formatting
# ---------------------------------------------------------------------------

class TestParseRtf:
    def test_plain_text(self):
        paras = parse_rtf(_rtf(r"\pard hello world\par"))
        assert len(paras) == 1
        assert paras[0][0]["text"] == "hello world"
        assert paras[0][0]["is_bold"] is False
        assert paras[0][0]["is_italic"] is False

    def test_bold_on_off(self):
        paras = parse_rtf(_rtf(r"\pard \b bold text\b0  normal\par"))
        assert len(paras) == 1
        spans = paras[0]
        bold_span = next(s for s in spans if s["is_bold"])
        assert "bold" in bold_span["text"]
        normal_span = next(s for s in spans if not s["is_bold"])
        assert "normal" in normal_span["text"]

    def test_italic_on_off(self):
        paras = parse_rtf(_rtf(r"\pard \i italic\i0  plain\par"))
        spans = paras[0]
        assert any(s["is_italic"] for s in spans)
        assert any(not s["is_italic"] for s in spans)

    def test_font_size_half_points(self):
        # \fs28 = 14 pt
        paras = parse_rtf(_rtf(r"\pard \fs28 big text\par"))
        assert paras[0][0]["size"] == pytest.approx(14.0)

    def test_plain_resets_formatting(self):
        paras = parse_rtf(_rtf(r"\pard \b\fs28 bold big\plain  normal\par"))
        spans = paras[0]
        normal = next(s for s in spans if "normal" in s["text"])
        assert normal["is_bold"] is False
        assert normal["size"] == pytest.approx(12.0)

    def test_multiple_paragraphs(self):
        paras = parse_rtf(_rtf(r"\pard first\par second\par third\par"))
        texts = ["".join(s["text"] for s in p) for p in paras]
        assert "first" in texts[0]
        assert "second" in texts[1]
        assert "third" in texts[2]

    def test_line_break_creates_paragraph(self):
        paras = parse_rtf(_rtf(r"\pard line one\line line two\par"))
        assert len(paras) == 2

    def test_group_inherits_formatting(self):
        # Bold set in outer group, inner group should inherit it.
        paras = parse_rtf(_rtf(r"\pard \b outer{  inner}\par"))
        for span in paras[0]:
            assert span["is_bold"] is True

    def test_group_scoped_formatting(self):
        # Bold set only inside a group should not affect text outside.
        paras = parse_rtf(_rtf(r"\pard before{\b  bold}  after\par"))
        spans = paras[0]
        before = next(s for s in spans if "before" in s["text"])
        after = next(s for s in spans if "after" in s["text"])
        bold = next(s for s in spans if "bold" in s["text"])
        assert before["is_bold"] is False
        assert bold["is_bold"] is True
        assert after["is_bold"] is False

    def test_hex_byte_decoding(self):
        # \'e9 = é in cp1252
        paras = parse_rtf(_rtf(r"\pard caf\'e9\par"))
        text = "".join(s["text"] for s in paras[0])
        assert "café" in text

    def test_skips_fonttbl(self):
        rtf = _rtf(r"{\fonttbl\f0\fswiss Helvetica;}\pard text\par")
        paras = parse_rtf(rtf)
        text = "".join(s["text"] for p in paras for s in p)
        assert "Helvetica" not in text
        assert "text" in text

    def test_skips_optional_destination(self):
        # {\* \unknown content} should be skipped
        rtf = _rtf(r"{\*\expandedcolortbl ;;\red255\green0\blue0;}\pard visible\par")
        paras = parse_rtf(rtf)
        text = "".join(s["text"] for p in paras for s in p)
        assert "visible" in text
        assert "red" not in text

    def test_escaped_braces(self):
        paras = parse_rtf(_rtf(r"\pard \{literal braces\}\par"))
        text = "".join(s["text"] for s in paras[0])
        assert "{" in text
        assert "}" in text

    def test_empty_rtf(self):
        paras = parse_rtf(b"{\\rtf1}")
        assert paras == []

    def test_no_par_still_returns_content(self):
        # Content with no \par should be returned on flush at end.
        paras = parse_rtf(_rtf(r"\pard some text"))
        assert len(paras) == 1
        assert "some text" in paras[0][0]["text"]


# ---------------------------------------------------------------------------
# parse_rtf — Arabic Unicode (Apple's \uN? encoding)
# ---------------------------------------------------------------------------

class TestArabicUnicodeExpansion:
    # RTF encodes non-ASCII as \uN? where N is the DECIMAL codepoint.
    # ك = U+0643 hex = 1603 decimal → RTF escape: \u1603?
    # ت = U+062A hex = 1578 decimal → RTF escape: \u1578?
    # ب = U+0628 hex = 1576 decimal → RTF escape: \u1576?
    # In Python raw strings, r"\u1603?" is the literal 8 chars: \ u 1 6 0 3 ?

    def test_arabic_via_u_escape_question_mark(self):
        paras = parse_rtf(_rtf(r"\pard \u1603?\par"))
        text = "".join(s["text"] for s in paras[0])
        assert "ك" in text

    def test_arabic_via_u_escape_hex_replacement(self):
        # Replacement is \'3f ('?') instead of bare '?'
        paras = parse_rtf(_rtf(r"\pard \u1603\'3f\par"))
        text = "".join(s["text"] for s in paras[0])
        assert "ك" in text

    def test_arabic_word_ktb(self):
        paras = parse_rtf(_rtf(r"\pard \u1603?\u1578?\u1576?\par"))
        text = "".join(s["text"] for s in paras[0])
        assert "كتب" in text

    def test_bold_arabic_headword(self):
        paras = parse_rtf(_rtf(r"\pard \b \u1603?\u1578?\u1576?\b0  to write\par"))
        spans = paras[0]
        bold = next(s for s in spans if s["is_bold"])
        assert "كتب" in bold["text"]
        normal = next(s for s in spans if not s["is_bold"])
        assert "to write" in normal["text"]

    def test_negative_u_param(self):
        # RTF uses signed 16-bit; U+FB50 = 64336.  As signed 16-bit: 64336 - 65536 = -1200.
        # _expand_rtf_unicode: int("-1200") % 65536 = 64336 → chr(64336) = U+FB50 (ﭐ)
        paras = parse_rtf(_rtf(r"\pard \u-1200?\par"))
        text = "".join(s["text"] for s in paras[0])
        assert chr(64336) in text


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    """detect_language() must return 'ar' or 'en' and never raise."""

    def test_pure_arabic_script(self):
        assert detect_language("كتب") == "ar"

    def test_pure_latin_word(self):
        assert detect_language("hello") == "en"

    def test_arabic_headword_with_translit(self):
        # >30% Arabic chars → fast path, no library needed
        assert detect_language("كتب kataba") == "ar"

    def test_english_definition(self):
        # Long enough for langdetect to identify as English
        assert detect_language("to write a book or compose a letter") == "en"

    def test_empty_string(self):
        assert detect_language("") == "en"

    def test_whitespace_only(self):
        assert detect_language("   ") == "en"

    def test_digits_only(self):
        assert detect_language("1234") == "en"

    def test_arabic_with_diacritics(self):
        # Diacritical marks (U+064B-U+065F) are in the Arabic block
        assert detect_language("كَتَبَ") == "ar"

    def test_short_abbreviation_defaults_english(self):
        # "n." is too short for langdetect; should default to 'en'
        assert detect_language("n.") == "en"

    def test_return_is_ar_or_en(self):
        for text in ["كتب", "hello world", "", "kitāb", "مرحبا"]:
            result = detect_language(text)
            assert result in ("ar", "en"), f"Unexpected language code {result!r} for {text!r}"


# ---------------------------------------------------------------------------
# paragraphs_to_page_json
# ---------------------------------------------------------------------------

class TestParagraphsToPageJson:
    def _sample_paras(self):
        return [
            [
                {"text": "كتب", "is_bold": True, "is_italic": False, "size": 14.0},
                {"text": " kataba", "is_bold": False, "is_italic": False, "size": 14.0},
            ],
            [
                {"text": "n.", "is_bold": False, "is_italic": True, "size": 12.0},
                {"text": " book", "is_bold": False, "is_italic": False, "size": 12.0},
            ],
        ]

    def test_schema_keys(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        assert result["page_number"] == 42
        assert result["source"] == "rtf_ocr"
        assert "blocks" in result

    def test_span_font_is_rtf_ocr(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        for block in result["blocks"]:
            for line in block["lines"]:
                for span in line["spans"]:
                    assert span["font"] == "rtf-ocr"

    def test_bold_italic_preserved(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        spans = [
            s
            for b in result["blocks"]
            for ln in b["lines"]
            for s in ln["spans"]
        ]
        bold = next(s for s in spans if s["is_bold"])
        assert bold["text"] == "كتب"
        italic = next(s for s in spans if s["is_italic"])
        assert italic["text"] == "n."

    def test_is_arabic_flag(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        spans = {
            s["text"]: s
            for b in result["blocks"]
            for ln in b["lines"]
            for s in ln["spans"]
        }
        assert spans["كتب"]["is_arabic"] is True
        assert spans["kataba"]["is_arabic"] is False

    def test_language_field_set(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        spans = {
            s["text"]: s
            for b in result["blocks"]
            for ln in b["lines"]
            for s in ln["spans"]
        }
        assert spans["كتب"]["language"] == "ar"
        assert spans["kataba"]["language"] == "en"

    def test_column_set_by_language(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        spans = {
            s["text"]: s
            for b in result["blocks"]
            for ln in b["lines"]
            for s in ln["spans"]
        }
        assert spans["كتب"]["column"] == "arabic"
        assert spans["kataba"]["column"] == "english"

    def test_column_values_are_arabic_or_english(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        for b in result["blocks"]:
            for ln in b["lines"]:
                for s in ln["spans"]:
                    assert s["column"] in ("arabic", "english")

    def test_language_values_are_ar_or_en(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        for b in result["blocks"]:
            for ln in b["lines"]:
                for s in ln["spans"]:
                    assert s["language"] in ("ar", "en")

    def test_bbox_zeroed(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        for block in result["blocks"]:
            assert block["bbox"] == [0.0, 0.0, 0.0, 0.0]
            for line in block["lines"]:
                assert line["bbox"] == [0.0, 0.0, 0.0, 0.0]
                for span in line["spans"]:
                    assert span["bbox"] == [0.0, 0.0, 0.0, 0.0]

    def test_flags_bitmask(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        spans = {
            s["text"]: s
            for b in result["blocks"]
            for ln in b["lines"]
            for s in ln["spans"]
        }
        assert spans["كتب"]["flags"] == 16       # bold = bit 4
        assert spans["n."]["flags"] == 2         # italic = bit 1
        assert spans["book"]["flags"] == 0       # normal

    def test_empty_paragraphs_yields_empty_blocks(self):
        result = paragraphs_to_page_json([], page_number=1)
        assert result["blocks"] == []

    def test_roundtrip_json_serialisable(self):
        result = paragraphs_to_page_json(self._sample_paras(), page_number=42)
        # Should not raise
        json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# _infer_page_number
# ---------------------------------------------------------------------------

class TestInferPageNumber:
    def test_page_nnnn(self):
        assert _infer_page_number("page_0042") == 42

    def test_page_n(self):
        assert _infer_page_number("page_1") == 1

    def test_page_dash_n(self):
        assert _infer_page_number("page-100") == 100

    def test_plain_number(self):
        assert _infer_page_number("0042") == 42

    def test_page_no_separator(self):
        assert _infer_page_number("page42") == 42

    def test_uppercase(self):
        assert _infer_page_number("PAGE_42") == 42

    def test_unrecognised(self):
        assert _infer_page_number("something_else") is None

    def test_no_number(self):
        assert _infer_page_number("page") is None
