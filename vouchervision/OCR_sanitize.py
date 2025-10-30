# OCR_Sanitize.py
import re, json
from typing import Any
from openpyxl.cell.cell import TYPE_STRING

DANGER_PREFIXES = ("=", "+", "-", "@")

_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
# ---------------------------------------------------------------------
# Header patterns to remove from top-of-output OCR blocks
# (e.g. "gemini-2.5-flash OCR:", "Qwen2VL OCR:", "Florence2 OCR:", etc.)
# ---------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"(?im)^\s*(?:"                       # start of line, optional whitespace
    r"gemini[-\s]?(?:[0-9.]+)?(?:[-\s]?(?:flash|pro))?"
    r"|qwen2vl"
    r"|qwen[-\s]?(?:vl|vision|ocr|vlm)?"
    r"|florence[-\s]?(?:2|ocr)?"
    r"|deepseek[-\s]?(?:vl|ocr|vision)?"
    r"|google[-\s]?(?:cloud|vision|ocr)?"
    r"|pali[-\s]?(?:gemma|gemma2|gemini)?"
    r"|gpt[-\s]?(?:4|4v|4o|4o-mini)?"
    r"|ocr\s+version\s+\d+[:]?"           # OCR Version {i}:
    r")\s*\n?",                           # swallow trailing whitespace/newline
    re.IGNORECASE | re.MULTILINE,
)

def strip_headers(text: str) -> str:
    """Remove any leading engine headers like 'Gemini-2.5-Flash OCR:' etc."""
    if not isinstance(text, str):
        return text
    return _HEADER_RE.sub("", text)

def basic_cleanup(text: str) -> str:
    """
    Keep the content, but make it API/Excel-friendly:
    - normalize newlines -> space
    - drop §, «, »
    - remove visual wrappers <<< >>> ~~~
    - collapse multiple spaces
    """
    if not isinstance(text, str):
        return text
    s = text.replace("\r", " ").replace("\n", " ")
    s = s.replace("§", "").replace("«", "").replace("»", "")
    s = re.sub(r"(?:<{3,}|>{3,}|~{3,})", "", s)
    s = s.replace("~", "").replace("<", "").replace(">", "")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def excel_guard(text: str) -> str:
    """
    Prevent Excel from interpreting the cell as a formula
    when later exported to .xlsx.
    """
    if not isinstance(text, str) or not text:
        return text
    if text.startswith("'"):
        return text  # already guarded
    stripped = text.lstrip()
    if stripped and stripped[0] in DANGER_PREFIXES:
        lead = len(text) - len(stripped)
        return text[:lead] + "'" + stripped
    return text

def sanitize_for_storage(
    text: str,
    *,
    remove_headers: bool = True,
    guard_excel: bool = True,
) -> str:
    """
    One-shot sanitizer for the saved copy.
    Default behavior:
      - removes engine headers ("... OCR:")
      - cleans specials/wrappers
      - protects against Excel formula injection if later exported
    """
    if not isinstance(text, str):
        return text
    if remove_headers:
        text = strip_headers(text)
    text = basic_cleanup(text)
    if guard_excel:
        text = excel_guard(text)
    return text

def _basic_cleanup_scalar(s: str) -> str:
    """Remove newlines, special marks, wrapper tokens; collapse spaces."""
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace("§", "").replace("«", "").replace("»", "")
    s = re.sub(r"(?:<{3,}|>{3,}|~{3,})", "", s)
    s = s.replace("~", "").replace("<", "").replace(">", "")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _excel_guard_scalar(s: str) -> str:
    """Prevent Excel from interpreting cell as a formula."""
    if not s:
        return s
    if s.startswith("'"):
        return s
    stripped = s.lstrip()
    if stripped and stripped[0] in DANGER_PREFIXES:
        lead = len(s) - len(stripped)
        return s[:lead] + "'" + stripped
    return s

def _sanitize_xml_illegal(s: str) -> str:
    """Remove illegal XML control chars and normalize newlines to spaces."""
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return _XML_ILLEGAL.sub(" ", s)

def _flatten_for_excel(x):
    """
    Deterministic coercion:
      - str -> keep
      - list/tuple -> '|' join of str(x_i)
      - dict -> JSON (stable)
      - other scalars -> keep as-is (numbers stay numeric)
    """
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " | ".join(map(lambda v: str(v) if not isinstance(v, str) else v, x))
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    return x  # ints/floats/bool/None as-is

def to_excel_literal(value):
    """
    Final, write-time sanitizer:
      - flatten
      - convert to str if string-like payload
      - strip XML-illegal
      - guard against formula
      - cap extreme length (Excel cell limit 32,767 chars)
    """
    MAX_LEN = 32767

    v = _flatten_for_excel(value)

    # Only strings need guarding / XML cleanup
    if isinstance(v, str):
        v = _sanitize_xml_illegal(v)
        v = _excel_guard_scalar(v)
        if len(v) > MAX_LEN:
            v = v[:MAX_LEN - 1] + "…"

    return v

def write_excel_safe(sheet, row, col, value):
    """
    Write to a cell with defensive settings for strings.
    """
    v = to_excel_literal(value)
    cell = sheet.cell(row=row, column=col, value=v)

    # If it's a string, force Excel to treat it as text
    if isinstance(v, str):
        cell.data_type = TYPE_STRING   # ensure string type in the xlsx
        cell.number_format = "@"       # Excel 'Text' format
    return cell

def _clean_scalar_for_excel(x: Any) -> Any:
    """Combine all scalar cleanups + header stripping."""
    if isinstance(x, str):
        x = strip_headers(x)
        x = _sanitize_xml_illegal(x)
        x = _basic_cleanup_scalar(x)
        x = _excel_guard_scalar(x)
    return x

def sanitize_excel_record(obj: Any) -> Any:
    """
    Deep-sanitize a JSON-like object (dict/list/tuple/str):
      - remove OCR headers (Gemini, Qwen, etc.)
      - remove illegal XML characters
      - normalize newlines → spaces
      - remove § « » <<< >>> ~~~
      - collapse multiple spaces
      - guard against Excel formula injection (= + - @)
    """
    if isinstance(obj, dict):
        return {k: sanitize_excel_record(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_excel_record(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_excel_record(v) for v in obj)
    return _clean_scalar_for_excel(obj)




# -----------------------------
# Markdown → Plain Text (flat)
# -----------------------------
_MD_FENCE_RE = re.compile(r"(?s)```.*?```")  # fenced code blocks
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")  # `inline`
_MD_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")  # #, ##, ...
_MD_SETEXT_RE = re.compile(r"(?m)^\s{0,3}(?:=+|-+)\s*$")  # underlines after headings
_MD_BLOCKQUOTE_RE = re.compile(r"(?m)^\s{0,3}>\s?")  # >
_MD_TASKLIST_RE = re.compile(r"(?m)^\s{0,3}[-*+]\s+\[(?: |x|X)\]\s+")  # - [ ] / - [x]
_MD_BULLET_RE = re.compile(r"(?m)^\s{0,3}[-*+]\s+")  # - foo / * foo / + foo
_MD_NUM_LIST_RE = re.compile(r"(?m)^\s{0,3}\d{1,3}[.)]\s+")  # 1. foo / 1) foo
_MD_IMG_RE = re.compile(r"!\[([^\]]*)\]\((?:[^)]+)\)")  # ![alt](url) -> alt
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")  # [text](url) -> text
_MD_REF_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")  # [text][id] -> text
_MD_REF_DEF_RE = re.compile(r"(?m)^\s*\[[^\]]+\]:\s+\S+.*$")  # [id]: url -> remove line
_MD_AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>")  # <http://...> -> url
_MD_EMPH_RE = re.compile(r"(\*\*\*|___|\*\*|__|\*|_)(.+?)\1")  # emphasis -> inner
_MD_TABLE_SEP_RE = re.compile(r"(?m)^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")  # |---|---|
_MD_TABLE_PIPE_RE = re.compile(r"(?m)\s*\|\s*")  # collapse pipes to spaces
_MD_FOOTNOTE_MARK_RE = re.compile(r"\[\^[^\]]+\]")  # [^1]
_MD_HTML_TAG_RE = re.compile(r"</?[^>]+>")  # strip simple HTML tags
_MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>])")  # remove MD backslash escapes

def markdown_to_plaintext(text: str) -> str:
    """
    Convert Markdown to a single plain-text string:
      - remove all markdown fences, headings, lists, blockquotes
      - resolve links/images to their visible text
      - drop footnote markers and reference definitions
      - strip tables' pipes and separators
      - remove inline code backticks and emphasis markers
      - remove simple HTML tags
      - normalize whitespace to a single space
    """
    if not isinstance(text, str):
        return text

    s = text

    # Remove reference-style link definitions early (entire lines)
    s = _MD_REF_DEF_RE.sub(" ", s)

    # Remove fenced code blocks completely
    s = _MD_FENCE_RE.sub(" ", s)

    # Replace images/links with their visible text
    s = _MD_IMG_RE.sub(r"\1", s)
    s = _MD_LINK_RE.sub(r"\1", s)
    s = _MD_REF_LINK_RE.sub(r"\1", s)
    s = _MD_AUTOLINK_RE.sub(r"\1", s)

    # Strip inline code markers, keep inner text
    s = _MD_INLINE_CODE_RE.sub(r"\1", s)

    # Remove headings markers (#...) and setext underlines
    s = _MD_HEADING_RE.sub("", s)
    s = _MD_SETEXT_RE.sub(" ", s)

    # Remove blockquote and list markers (bullets, task, numbered)
    s = _MD_BLOCKQUOTE_RE.sub("", s)
    s = _MD_TASKLIST_RE.sub("", s)
    s = _MD_BULLET_RE.sub("", s)
    s = _MD_NUM_LIST_RE.sub("", s)

    # Tables: drop separator lines, replace pipes with spaces
    s = _MD_TABLE_SEP_RE.sub(" ", s)
    s = _MD_TABLE_PIPE_RE.sub(" ", s)

    # Emphasis: remove markers but keep inner text (run twice to catch nesting)
    for _ in range(2):
        s = _MD_EMPH_RE.sub(r"\2", s)

    # Footnote markers
    s = _MD_FOOTNOTE_MARK_RE.sub("", s)

    # Simple HTML tags
    s = _MD_HTML_TAG_RE.sub(" ", s)

    # Remove MD escape backslashes
    s = _MD_ESCAPE_RE.sub(r"\1", s)

    # Normalize whitespace (also collapse newlines/tabs)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def markdown_to_simple_text(
    text: str,
    *,
    remove_headers: bool = True,
    guard_excel: bool = False,
) -> str:
    """
    High-level helper tailored for your OCR pipeline:
      1) (Optional) remove leading engine headers like 'Gemini-2.5-Flash OCR:'
      2) strip all Markdown syntax to visible text only
      3) hard-normalize to a single flat string with spaces
      4) (Optional) Excel guard for '=' '+' '-' '@'

    Returns a single-line plain string.
    """
    if not isinstance(text, str):
        return text

    s = text
    if remove_headers:
        s = strip_headers(s)

    s = markdown_to_plaintext(s)

    # Final pass: remove your special wrapper tokens & illegal XML, collapse spaces
    s = _sanitize_xml_illegal(s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    if guard_excel:
        s = _excel_guard_scalar(s)

    return s
