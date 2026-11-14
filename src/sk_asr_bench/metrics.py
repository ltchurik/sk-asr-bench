"""Edit-distance metrics: WER, CER and diacritics error rate.

Implemented from scratch (no jiwer dependency) so the scoring is small,
auditable and identical on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from .normalize import NumberHook, normalize, strip_diacritics

# Alignment operations
EQ, SUB, DEL, INS = "=", "S", "D", "I"


@dataclass
class Alignment:
    """Result of aligning a reference against a hypothesis."""

    ops: List[Tuple[str, Optional[str], Optional[str]]] = field(default_factory=list)

    @property
    def substitutions(self) -> int:
        return sum(1 for op, _, _ in self.ops if op == SUB)

    @property
    def deletions(self) -> int:
        return sum(1 for op, _, _ in self.ops if op == DEL)

    @property
    def insertions(self) -> int:
        return sum(1 for op, _, _ in self.ops if op == INS)

    @property
    def hits(self) -> int:
        return sum(1 for op, _, _ in self.ops if op == EQ)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def ref_len(self) -> int:
        return self.hits + self.substitutions + self.deletions


def align(ref: Sequence[str], hyp: Sequence[str]) -> Alignment:
    """Levenshtein alignment with backtrace (unit costs).

    Ties are broken in the order match/substitution, deletion, insertion,
    which keeps substitutions paired word-to-word -- needed to detect
    diacritics-only errors.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prev[j - 1] + cost, prev[j] + 1, row[j - 1] + 1)

    ops: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                ops.append((EQ if cost == 0 else SUB, ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((DEL, ref[i - 1], None))
            i -= 1
        else:
            ops.append((INS, None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return Alignment(ops)


def _ratio(errors: int, total: int) -> float:
    if total == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / total


def wer(ref: str, hyp: str, number_hook: Optional[NumberHook] = None) -> float:
    """Word error rate of one utterance after normalization."""
    a = align(normalize(ref, number_hook).split(), normalize(hyp, number_hook).split())
    return _ratio(a.errors, a.ref_len)


def cer(ref: str, hyp: str, number_hook: Optional[NumberHook] = None) -> float:
    """Character error rate of one utterance (spaces count as characters)."""
    a = align(list(normalize(ref, number_hook)), list(normalize(hyp, number_hook)))
    return _ratio(a.errors, a.ref_len)


def diacritic_substitutions(a: Alignment) -> int:
    """Substitutions where words are identical once diacritics are removed."""
    return sum(
        1
        for op, r, h in a.ops
        if op == SUB and r is not None and h is not None and strip_diacritics(r) == strip_diacritics(h)
    )


def diacritics_error_rate(ref: str, hyp: str, number_hook: Optional[NumberHook] = None) -> float:
    """Share of reference words that are wrong *only* in diacritics.

    "Učiteľ" transcribed as "ucitel" is one diacritics error; "učiteľ" as
    "učiteľka" is a regular substitution and does not count here.
    """
    a = align(normalize(ref, number_hook).split(), normalize(hyp, number_hook).split())
    return _ratio(diacritic_substitutions(a), a.ref_len)


@dataclass
class CorpusScore:
    utterances: int
    ref_words: int
    ref_chars: int
    wer: float
    cer: float
    diacritics_error_rate: float
    substitutions: int
    deletions: int
    insertions: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def score_corpus(
    pairs: Iterable[Tuple[str, str]], number_hook: Optional[NumberHook] = None
) -> CorpusScore:
    """Corpus-level scores: total errors / total reference units (not a mean of per-utterance rates)."""
    n = words = chars = w_err = c_err = d_err = s = d = ins = 0
    for ref, hyp in pairs:
        n += 1
        r, h = normalize(ref, number_hook), normalize(hyp, number_hook)
        wa = align(r.split(), h.split())
        ca = align(list(r), list(h))
        words += wa.ref_len
        chars += ca.ref_len
        w_err += wa.errors
        c_err += ca.errors
        d_err += diacritic_substitutions(wa)
        s, d, ins = s + wa.substitutions, d + wa.deletions, ins + wa.insertions
    return CorpusScore(
        utterances=n,
        ref_words=words,
        ref_chars=chars,
        wer=_ratio(w_err, words),
        cer=_ratio(c_err, chars),
        diacritics_error_rate=_ratio(d_err, words),
        substitutions=s,
        deletions=d,
        insertions=ins,
    )
