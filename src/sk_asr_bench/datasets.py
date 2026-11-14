"""Test-set loaders.

Every loader yields :class:`Utterance` objects. Audio is never stored in this
repository; it is read from local files or fetched by the Hugging Face
``datasets`` library (``pip install "sk-asr-bench[datasets]"``).

Dataset specs accepted by the CLI:

- ``jsonl:PATH`` -- local manifest, one JSON object per line:
  ``{"id": "...", "text": "reference, may contain {num:...} tags", "audio": "clip.wav"}``
  (audio paths are relative to the manifest; ``audio`` is optional when you
  only score precomputed hypotheses)
- ``fleurs`` -- FLEURS Slovak test split (``google/fleurs``, config ``sk_sk``, CC-BY 4.0)
- ``commonvoice:DIR`` -- an extracted Common Voice Slovak release directory
  (``test.tsv`` + ``clips/``). Common Voice is distributed by Mozilla, not via
  this tool; download it yourself and accept its terms.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

Audio = Union[str, dict, None]  # path, {"array": ..., "sampling_rate": int}, or None


@dataclass
class Utterance:
    id: str
    text: str  # reference, optionally with {kind:...} entity tags
    audio: Audio = None


def _jsonl(path: str) -> Iterator[Utterance]:
    manifest = Path(path)
    with manifest.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio = row.get("audio")
            if isinstance(audio, str) and not Path(audio).is_absolute():
                audio = str(manifest.parent / audio)
            yield Utterance(id=str(row.get("id", n)), text=row["text"], audio=audio)


def _fleurs(split: str = "test") -> Iterator[Utterance]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit('FLEURS needs: pip install "sk-asr-bench[datasets]"') from exc
    ds = load_dataset("google/fleurs", "sk_sk", split=split, trust_remote_code=True)
    for row in ds:
        yield Utterance(id=f"fleurs-{row['id']}", text=row["raw_transcription"], audio=row["audio"])


def _commonvoice(directory: str, split: str = "test") -> Iterator[Utterance]:
    root = Path(directory)
    tsv = root / f"{split}.tsv"
    if not tsv.exists():
        raise SystemExit(f"{tsv} not found -- point commonvoice: at an extracted Common Voice 'sk' directory")
    with tsv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            yield Utterance(id=f"cv-{Path(row['path']).stem}", text=row["sentence"], audio=str(root / "clips" / row["path"]))


def load(spec: str, limit: Optional[int] = None) -> Iterator[Utterance]:
    """Load a dataset by spec (see module docstring)."""
    kind, _, arg = spec.partition(":")
    if kind == "jsonl" and arg:
        it: Iterator[Utterance] = _jsonl(arg)
    elif kind == "fleurs":
        it = _fleurs(arg or "test")
    elif kind == "commonvoice" and arg:
        it = _commonvoice(arg)
    else:
        raise SystemExit(f"unknown dataset spec {spec!r} (use jsonl:PATH, fleurs, commonvoice:DIR)")
    for n, utt in enumerate(it):
        if limit is not None and n >= limit:
            return
        yield utt
