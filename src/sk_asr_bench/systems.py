"""ASR systems under test.

System specs accepted by the CLI:

- ``hyp:PATH`` -- precomputed hypotheses, JSONL ``{"id": "...", "text": "..."}``.
  Lets anyone score output from a system this tool does not wrap.
- ``faster-whisper:MODEL`` -- local CTranslate2 Whisper, e.g. ``faster-whisper:large-v3``
  (``pip install "sk-asr-bench[whisper]"``)
- ``transformers:MODEL`` -- any Hugging Face ASR model, e.g.
  ``transformers:NaiveNeuron/whisper-large-v3-sk`` (``[transformers]`` extra)
- ``openai:MODEL`` -- any OpenAI-compatible ``/audio/transcriptions`` endpoint
  (OpenAI, Groq, a local vLLM/speaches server…). Uses ``OPENAI_BASE_URL``
  (default ``https://api.openai.com/v1``) and ``OPENAI_API_KEY``.

Planned, not implemented yet: ``gemini``, ``elevenlabs``, ``azure``, ``google``.
"""

from __future__ import annotations

import io
import json
import os
import struct
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Callable, Dict, Iterable

from .datasets import Audio, Utterance

TARGET_SR = 16_000

Transcriber = Callable[[Utterance], str]


def _audio_array(audio: Audio):
    """Return (float32 numpy array, sampling rate 16 kHz) for in-memory or file audio."""
    import numpy as np  # optional; installed with every model backend

    if isinstance(audio, dict):
        arr, sr = np.asarray(audio["array"], dtype="float32"), int(audio["sampling_rate"])
    elif isinstance(audio, str):
        import soundfile as sf  # type: ignore

        arr, sr = sf.read(audio, dtype="float32", always_2d=False)
    else:
        raise ValueError("utterance has no audio")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        import librosa  # type: ignore

        arr, sr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR), TARGET_SR
    return arr, sr


def wav_bytes(samples: Iterable[float], sampling_rate: int) -> bytes:
    """Encode mono float samples in [-1, 1] as 16-bit PCM WAV (stdlib only)."""
    pcm = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, float(s))) * 32767)) for s in samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sampling_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _hyp_file(path: str) -> Transcriber:
    hyps: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                hyps[str(row["id"])] = row["text"]

    def run(utt: Utterance) -> str:
        if utt.id not in hyps:
            raise KeyError(f"no hypothesis for utterance {utt.id!r} in {path}")
        return hyps[utt.id]

    return run


def _faster_whisper(model: str) -> Transcriber:  # pragma: no cover - needs model weights
    from faster_whisper import WhisperModel  # type: ignore

    m = WhisperModel(model, device="auto", compute_type="auto")

    def run(utt: Utterance) -> str:
        audio = utt.audio if isinstance(utt.audio, str) else _audio_array(utt.audio)[0]
        segments, _ = m.transcribe(audio, language="sk", beam_size=5)
        return " ".join(s.text.strip() for s in segments)

    return run


def _transformers(model: str) -> Transcriber:  # pragma: no cover - needs model weights
    from transformers import pipeline  # type: ignore

    asr = pipeline("automatic-speech-recognition", model=model, device_map="auto")
    kwargs = {"generate_kwargs": {"language": "slovak", "task": "transcribe"}} if "whisper" in model.lower() else {}

    def run(utt: Utterance) -> str:
        arr, sr = _audio_array(utt.audio)
        return asr({"raw": arr, "sampling_rate": sr}, chunk_length_s=30, **kwargs)["text"].strip()

    return run


def _openai(model: str) -> Transcriber:  # pragma: no cover - needs network
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")

    def run(utt: Utterance) -> str:
        if isinstance(utt.audio, str):
            data, name = Path(utt.audio).read_bytes(), Path(utt.audio).name
        else:
            arr, sr = _audio_array(utt.audio)
            data, name = wav_bytes(arr, sr), "audio.wav"
        boundary = uuid.uuid4().hex
        parts = []
        for field, value in (("model", model), ("language", "sk"), ("response_format", "json")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            f"{base}/audio/transcriptions",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["text"].strip()

    return run


PLANNED = {"gemini", "elevenlabs", "azure", "google"}


def load(spec: str) -> Transcriber:
    kind, _, arg = spec.partition(":")
    if kind == "hyp" and arg:
        return _hyp_file(arg)
    if kind == "faster-whisper" and arg:
        return _faster_whisper(arg)
    if kind == "transformers" and arg:
        return _transformers(arg)
    if kind == "openai" and arg:
        return _openai(arg)
    if kind in PLANNED:
        raise SystemExit(f"system {kind!r} is planned but not implemented yet -- contributions welcome")
    raise SystemExit(
        f"unknown system spec {spec!r} (use hyp:PATH, faster-whisper:MODEL, transformers:MODEL, openai:MODEL)"
    )
