"""Tagged references for numeral and named-entity accuracy.

References can mark spans that matter most in practice::

    Platba {num:1 250} eur prišla {date:12. marca} od {name:Jána Nováka}.

The plain reference (tags removed) is used for WER/CER. Each tagged span is
then checked separately: it counts as correct when its normalized form
appears as a contiguous word sequence in the normalized hypothesis.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .normalize import NumberHook, normalize

TAG = re.compile(r"\{(?P<kind>[a-z_]+):(?P<text>[^{}]+)\}")


@dataclass(frozen=True)
class Entity:
    kind: str
    text: str


def parse_tagged(reference: str) -> Tuple[str, List[Entity]]:
    """Return (plain reference, entities) for a tagged reference string."""
    entities = [Entity(m.group("kind"), m.group("text")) for m in TAG.finditer(reference)]
    plain = TAG.sub(lambda m: m.group("text"), reference)
    return plain, entities


def _contains(haystack: List[str], needle: List[str]) -> bool:
    if not needle:
        return True
    k = len(needle)
    return any(haystack[i : i + k] == needle for i in range(len(haystack) - k + 1))


def entity_hit(entity: Entity, hypothesis: str, number_hook: Optional[NumberHook] = None) -> bool:
    return _contains(normalize(hypothesis, number_hook).split(), normalize(entity.text, number_hook).split())


def entity_accuracy(
    pairs: Iterable[Tuple[str, str]], number_hook: Optional[NumberHook] = None
) -> Dict[str, Dict[str, float]]:
    """Accuracy per entity kind over (tagged reference, hypothesis) pairs.

    Returns ``{kind: {"total": n, "correct": k, "accuracy": k / n}}``.
    Kinds with no occurrences are omitted.
    """
    total: Dict[str, int] = defaultdict(int)
    correct: Dict[str, int] = defaultdict(int)
    for ref, hyp in pairs:
        for ent in parse_tagged(ref)[1]:
            total[ent.kind] += 1
            correct[ent.kind] += entity_hit(ent, hyp, number_hook)
    return {
        k: {"total": total[k], "correct": correct[k], "accuracy": correct[k] / total[k]}
        for k in sorted(total)
    }
