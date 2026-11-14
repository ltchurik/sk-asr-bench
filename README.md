# sk-asr-bench

**How well does AI understand spoken Slovak?** A reproducible, independent benchmark of Slovak speech recognition: open models (Whisper and its Slovak fine-tunes, NVIDIA Parakeet/Canary, …) and commercial APIs scored on the same test sets with the same normalization.

[![ci](https://github.com/ltchurik/sk-asr-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/ltchurik/sk-asr-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

> **Status: 🚧 Early development — no published results yet.** The scoring core and CLI work and are tested; the first leaderboard run is in progress. Numbers will appear here only once they are reproducible from this repository.

🇸🇰 [Po slovensky](#po-slovensky) · [Why](#why) · [Methodology](#methodology) · [Quickstart](#quickstart) · [Roadmap](#roadmap) · [Citing](#citing)

Part of a series on Slovak in AI: [sk-llm-eval](https://github.com/ltchurik/sk-llm-eval) (does the model *write* correct Slovak?) and [sk-tokenizer-bench](https://github.com/ltchurik/sk-tokenizer-bench) (how much more does Slovak cost in tokens?).

## Why

If you want to transcribe Slovak meetings, calls, court hearings or podcasts today, you have to choose between a dozen ASR systems based on numbers that cannot be compared:

- **Different test sets.** FLEURS, Common Voice (several versions), parliament recordings, children's speech.
- **Different metrics.** Some report WER, some only CER; normalization (case, punctuation, numbers) is rarely stated.
- **Self-reported.** Most Slovak figures come from the authors of a fine-tune or from a vendor's marketing page.

Nobody runs all the systems on the same audio with the same scorer. This repository does, and publishes everything needed to reproduce it.

## What it measures

| Metric | What it tells you |
|---|---|
| **WER** | Word error rate after normalization (case, punctuation, whitespace). Pooled over the corpus, not averaged per utterance. |
| **CER** | Character error rate; fairer for a highly inflected language where one wrong suffix makes the whole word wrong. |
| **Diacritics error rate** | Share of words that are wrong *only* in diacritics (`ucitel` vs `učiteľ`). Diacritics are never stripped by the normalizer, because a Slovak reader sees them as errors. |
| **Numbers, dates, names** | Accuracy on tagged spans in the reference (`{num:1 250}`, `{date:12. marca}`, `{name:Jána Nováka}`): the parts of a transcript that matter most in practice and that WER hides. |
| **Speed** | Transcription time per run (real-time factor on the reference hardware: NVIDIA DGX Spark, GB10). |

## Methodology

- **Normalization** (`sk_asr_bench.normalize`): Unicode NFC, lowercase, punctuation and symbols removed except inside numbers (`3,5`, `10:30`), thousands separators joined (`1 250` → `1250`), whitespace collapsed. Optional hook for number verbalization. Diacritics are kept.
- **Alignment**: plain Levenshtein with backtrace, implemented in ~100 lines with no dependencies, so the scorer is auditable and identical everywhere.
- **Test sets** (audio is never stored in this repo; loaders fetch it):
  - [FLEURS](https://huggingface.co/datasets/google/fleurs) `sk_sk` test split (CC-BY 4.0)
  - [Common Voice](https://commonvoice.mozilla.org) Slovak test split (CC0), downloaded by you from Mozilla
  - a **new held-out set** written and recorded for this benchmark (numbers, dates, names, regional accents, telephone-quality audio), so that no model can have trained on it. *Planned.*
  - SloPalSpeech (parliamentary speech) will be added only after its dataset licence is confirmed to allow evaluation use (the paper is CC BY-NC-ND).
- **Systems**: run with default settings and `language=sk`; every run stores per-utterance transcripts so that each number can be inspected.

## Prior art

- **SloPalSpeech** ([arXiv 2509.19270](https://arxiv.org/abs/2509.19270)): a 2,806-hour Slovak parliamentary corpus with Whisper fine-tunes and baselines on FLEURS and Common Voice. An excellent training resource; it evaluates its own models, not a broad set of systems.
- **Model cards** such as [NaiveNeuron/whisper-large-v3-sk](https://huggingface.co/NaiveNeuron/whisper-large-v3-sk) and [BuzzASR/slovak](https://huggingface.co/BuzzASR/slovak) report their own WER/CER on their chosen splits.
- **Vendors** (e.g. ElevenLabs) publish Slovak WER claims without a reproducible method.

sk-asr-bench does not compete with these; it puts them side by side under one scorer.

## Quickstart

What works today: scoring any system's output and running local or OpenAI-compatible models.

```bash
git clone https://github.com/ltchurik/sk-asr-bench
cd sk-asr-bench
pip install -e ".[dev]"
pytest

# score precomputed transcripts (any system, any source)
sk-asr-bench run --system hyp:tests/fixtures/hyp.jsonl --dataset jsonl:tests/fixtures/refs.jsonl

# local Whisper on FLEURS (first 50 utterances)
pip install -e ".[datasets,whisper]"
sk-asr-bench run --system faster-whisper:large-v3 --dataset fleurs --limit 50

# any Hugging Face ASR model
pip install -e ".[datasets,transformers]"
sk-asr-bench run --system transformers:NaiveNeuron/whisper-large-v3-sk --dataset fleurs

# any OpenAI-compatible /audio/transcriptions endpoint
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=... \
  sk-asr-bench run --system openai:whisper-1 --dataset fleurs

# leaderboard from all runs in results/
sk-asr-bench report
```

Runs are resumable: transcripts are cached in `results/<system>__<dataset>.jsonl`; rerunning skips finished utterances (`--fresh` starts over).

### Your own test set

A dataset is a JSONL manifest; tags mark spans scored separately:

```json
{"id": "call-001", "audio": "clips/call-001.wav", "text": "Platba {num:1 250} eur prišla {date:12. marca} od {name:Jána Nováka}."}
```

## Roadmap

- [x] Scoring core: normalization, WER, CER, diacritics error rate, entity accuracy
- [x] CLI with resumable runs and Markdown leaderboard
- [x] Adapters: precomputed transcripts, faster-whisper, Hugging Face transformers, OpenAI-compatible API
- [ ] First leaderboard on FLEURS + Common Voice (Whisper tiny→large-v3, Slovak fine-tunes, Parakeet/Canary)
- [ ] Adapters for Gemini, ElevenLabs, Azure Speech, Google Speech-to-Text
- [ ] Held-out Slovak test set: numbers, dates, names, accents, telephone audio
- [ ] Optional number verbalization for Slovak (`25` ↔ `dvadsaťpäť`)
- [ ] Hugging Face leaderboard Space and Zenodo DOI

Contributions are welcome: new systems, test audio you have rights to share, or fixes to the scorer.

## Po slovensky

**Ako dobre rozumie umelá inteligencia hovorenej slovenčine?** sk-asr-bench je nezávislý a reprodukovateľný benchmark rozpoznávania slovenskej reči. Otvorené modely (Whisper a jeho slovenské doladenia, NVIDIA Parakeet/Canary) aj komerčné API (Gemini, ElevenLabs, Azure, Google) porovnáva na rovnakých nahrávkach a rovnakým spôsobom hodnotenia.

Dnešné čísla pre slovenčinu sa nedajú porovnať: každý meria na inej sade, inou metrikou a väčšinou ich zverejňuje autor modelu alebo predajca. Benchmark meria chybovosť slov (WER) a znakov (CER), osobitne chyby v diakritike a presnosť čísel, dátumov a mien, teda častí prepisu, na ktorých v praxi záleží najviac. Pribudne aj vlastná testovacia sada s číslami, menami, nárečiami a telefonickou kvalitou zvuku, na ktorej sa žiadny model nemohol učiť.

**Stav:** v ranom vývoji, výsledky zatiaľ nie sú zverejnené. Hodnotiace jadro a CLI fungujú a sú otestované.

## Citing

See [CITATION.cff](CITATION.cff). A Zenodo DOI will be added with the first release.

## Author

[Ľubomír Tchurik](https://lubomirtchurik.sk) · [ORCID 0009-0005-8053-9405](https://orcid.org/0009-0005-8053-9405)

## License

Code: [MIT](LICENSE). Test sets keep their own licences (FLEURS CC-BY 4.0, Common Voice CC0).
