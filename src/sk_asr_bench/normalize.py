"""Text normalization used before scoring.

Scoring Slovak ASR fairly needs a normalizer that removes differences nobody
cares about (case, punctuation, spacing) but keeps the ones that matter:
diacritics are *never* stripped here, because a missing "ľ" or "ô" is a real
error for a Slovak reader. Diacritics are measured separately by
:func:`sk_asr_bench.metrics.diacritics_error_rate`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Optional

NumberHook = Callable[[str], str]

# Thousands separators inside numbers: "1 250", "1 250" (nbsp), "1 250" (narrow nbsp)
_THOUSANDS = re.compile(r"(?<=\d)[   ](?=\d{3}(?!\d))")
# Punctuation between two digits is part of the number ("3,5", "12.03.2026", "10:30")
_DIGIT_PUNCT = re.compile(r"(?<=\d)[.,:/](?=\d)")
_PROTECT = {".": "", ",": "", ":": "", "/": ""}
_UNPROTECT = {v: k for k, v in _PROTECT.items()}
_SPACES = re.compile(r"\s+")


def _is_dropped(ch: str) -> bool:
    cat = unicodedata.category(ch)
    # P* = punctuation, S* = symbols (quotes, dashes, €, % …); letters, marks, digits stay
    return cat.startswith("P") or cat.startswith("S")


def normalize(text: str, number_hook: Optional[NumberHook] = None) -> str:
    """Normalize a transcript for WER/CER scoring.

    - Unicode NFC (so "ô" typed as o + combining circumflex equals "ô")
    - lowercase
    - thousands separators inside numbers removed ("1 250" -> "1250")
    - punctuation and symbols removed, except inside numbers ("3,5", "10:30")
    - whitespace collapsed

    ``number_hook`` is an optional callable applied to the normalized string,
    e.g. to expand digits into words ("25" -> "dvadsaťpäť") so that systems
    writing numbers as words are not penalized. None is applied by default:
    the benchmark reports numbers as written and measures them separately.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = _THOUSANDS.sub("", text)
    text = _DIGIT_PUNCT.sub(lambda m: _PROTECT[m.group(0)], text)
    text = "".join(" " if _is_dropped(ch) else ch for ch in text)
    text = "".join(_UNPROTECT.get(ch, ch) for ch in text)
    text = _SPACES.sub(" ", text).strip()
    if number_hook is not None:
        text = _SPACES.sub(" ", number_hook(text)).strip()
    return text


def strip_diacritics(text: str) -> str:
    """Remove combining marks: "ľščťžýáíéďňôúäŕĺ" -> "lsctzyaiednouarl"."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)
