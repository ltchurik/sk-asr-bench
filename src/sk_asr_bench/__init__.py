"""sk-asr-bench: reproducible benchmark of Slovak speech recognition."""

from .metrics import Alignment, align, cer, diacritics_error_rate, score_corpus, wer
from .normalize import normalize, strip_diacritics
from .entities import Entity, entity_accuracy, parse_tagged

__version__ = "0.2.2"

__all__ = [
    "Alignment",
    "Entity",
    "align",
    "cer",
    "diacritics_error_rate",
    "entity_accuracy",
    "normalize",
    "parse_tagged",
    "score_corpus",
    "strip_diacritics",
    "wer",
]
