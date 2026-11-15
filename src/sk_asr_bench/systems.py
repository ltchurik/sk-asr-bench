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
# Whisper is decoded greedily (1 beam) by every adapter, like the CTC/TDT systems it
# is compared with. Beam search of 5 lowers Whisper large-v3 WER on FLEURS-sk by about
# one point but is several times slower; the transformers pipeline would otherwise
# silently default to 5 beams, so it is always set explicitly.
WHISPER_BEAMS = 1

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
        segments, _ = m.transcribe(audio, language="sk", beam_size=WHISPER_BEAMS)
        return " ".join(s.text.strip() for s in segments)

    return run


STANDARD_WHISPER_VOCABS = (51865, 51866)  # multilingual Whisper up to v2 / v3


def _fix_whisper_generation_config(m, model: str) -> None:  # pragma: no cover - needs model weights
    """Make ``generate(language=..., task=...)`` work for fine-tunes whose config lacks ``lang_to_id``.

    * Models with their own tokenizer (BuzzASR) do ship the special-token ids, but
      transformers ignores a generation_config.json marked ``_from_model_config``;
      copy the ids from the raw file -- the stock Whisper ids would be wrong there.
    * Old fine-tunes with the stock vocabulary (HF Whisper event 2022, e.g. mikr/*)
      predate language/task kwargs; borrow the config of the matching base model.
    """
    import json

    from huggingface_hub import hf_hub_download  # type: ignore
    from transformers import GenerationConfig  # type: ignore

    gc = m.generation_config
    try:
        raw = json.loads(open(hf_hub_download(model, "generation_config.json"), encoding="utf-8").read())
    except Exception:  # noqa: BLE001 - no file: fall through to the base model
        raw = {}
    if "lang_to_id" in raw:
        for key in ("lang_to_id", "task_to_id", "no_timestamps_token_id"):
            if key in raw:
                setattr(gc, key, raw[key])
        gc.is_multilingual = True
        return
    cfg = m.config
    if cfg.vocab_size not in STANDARD_WHISPER_VOCABS:
        raise ValueError(f"{model}: custom Whisper vocabulary without lang_to_id; cannot force Slovak")
    base = {4: "tiny", 6: "base", 12: "small", 24: "medium"}.get(cfg.encoder_layers)
    if base is None:
        base = "large-v3" if cfg.num_mel_bins == 128 else "large-v2"
    m.generation_config = GenerationConfig.from_pretrained(f"openai/whisper-{base}")


def _transformers(model: str) -> Transcriber:  # pragma: no cover - needs model weights
    import torch  # type: ignore
    from transformers import pipeline  # type: ignore

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    asr = pipeline("automatic-speech-recognition", model=model, device=0 if cuda else -1, dtype=dtype)
    whisper = asr.model.config.model_type == "whisper"
    if whisper and not hasattr(asr.model.generation_config, "lang_to_id"):
        _fix_whisper_generation_config(asr.model, model)
    kwargs = (
        {"generate_kwargs": {"language": "slovak", "task": "transcribe", "num_beams": WHISPER_BEAMS, "use_cache": True}} if whisper else {}
    )

    def run(utt: Utterance) -> str:
        arr, sr = _audio_array(utt.audio)
        return asr({"raw": arr, "sampling_rate": sr}, chunk_length_s=30, **kwargs)["text"].strip()

    if not whisper:
        return run

    # The pipeline does not batch Whisper when chunking is on, so clips that fit in
    # Whisper's 30 s window are decoded with model.generate in real batches; longer
    # clips still go through the pipeline's chunked long-form path one by one.
    def batch(utts):
        arrays = [_audio_array(u.audio)[0] for u in utts]
        short = [i for i, a in enumerate(arrays) if len(a) <= 30 * TARGET_SR]
        out = [""] * len(utts)
        if short:
            feats = asr.feature_extractor(
                [arrays[i] for i in short], sampling_rate=TARGET_SR, return_tensors="pt", return_attention_mask=True
            )
            with torch.inference_mode():
                ids = asr.model.generate(
                    feats.input_features.to(asr.model.device, dtype),
                    attention_mask=feats.attention_mask.to(asr.model.device),
                    language="sk",
                    task="transcribe",
                    num_beams=WHISPER_BEAMS,
                    # some fine-tunes (NaiveNeuron) ship use_cache=False: same output, but
                    # 3-4x slower and memory-hungry enough to OOM a 128 GB box at batch 32
                    use_cache=True,
                )
            for i, text in zip(short, asr.tokenizer.batch_decode(ids, skip_special_tokens=True)):
                out[i] = text.strip()
        for i in range(len(utts)):
            if i not in short:
                out[i] = run(utts[i])
        return out

    run.batch = batch  # type: ignore[attr-defined]
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
