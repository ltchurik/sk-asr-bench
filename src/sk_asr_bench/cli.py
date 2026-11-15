"""Command line interface: ``sk-asr-bench run`` and ``sk-asr-bench report``."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import __version__, datasets, systems
from .entities import entity_accuracy, parse_tagged
from .metrics import score_corpus


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _duration(audio) -> Optional[float]:
    """Length of the utterance audio in seconds, when it can be known cheaply."""
    if isinstance(audio, dict) and "array" in audio:
        return len(audio["array"]) / float(audio["sampling_rate"])
    if isinstance(audio, str):
        try:
            import soundfile as sf  # type: ignore

            return float(sf.info(audio).duration)
        except Exception:  # noqa: BLE001 - duration is informational only
            return None
    return None


def _dataset_label(spec: str) -> str:
    """Name a dataset by what it is, not where it sits on this machine (commonvoice:DIR -> commonvoice)."""
    return spec.split(":", 1)[0] if spec.startswith("commonvoice:") else spec


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = _dataset_label(args.dataset)
    stem = f"{_slug(args.system)}__{_slug(label)}"
    rows_path, summary_path = out / f"{stem}.jsonl", out / f"{stem}.summary.json"

    done = {}
    if rows_path.exists() and not args.fresh:  # resume long runs
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row

    transcribe = None
    rows = []
    pending: list = []

    def flush(fh) -> None:
        """Transcribe the pending utterances (one batch) and append their rows."""
        if not pending:
            return
        t0 = time.perf_counter()
        batch = getattr(transcribe, "batch", None)
        hyps = batch(pending) if batch and len(pending) > 1 else [transcribe(u) for u in pending]
        elapsed = time.perf_counter() - t0
        durs = [_duration(u.audio) for u in pending]
        known = sum(d for d in durs if d) or 0.0
        for utt, hyp, dur in zip(pending, hyps, durs):
            # Batch time is split across utterances in proportion to their audio length.
            share = (dur / known) if (dur and known) else 1.0 / len(pending)
            row = {"id": utt.id, "ref": utt.text, "hyp": hyp, "seconds": round(elapsed * share, 4)}
            if dur is not None:
                row["audio_seconds"] = round(dur, 3)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
        fh.flush()
        pending.clear()

    with rows_path.open("a" if done else "w", encoding="utf-8") as fh:
        for utt in datasets.load(args.dataset, limit=args.limit):
            if utt.id in done:
                rows.append(done[utt.id])
                continue
            if transcribe is None:
                transcribe = systems.load(args.system)
            pending.append(utt)
            if len(pending) >= args.batch_size:
                flush(fh)
        flush(fh)

    pairs = [(parse_tagged(r["ref"])[0], r["hyp"]) for r in rows]
    summary = {
        "system": args.system,
        "dataset": label,
        "limit": args.limit,
        "sk_asr_bench_version": __version__,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scores": score_corpus(pairs).as_dict(),
        "entities": entity_accuracy((r["ref"], r["hyp"]) for r in rows),
        "batch_size": args.batch_size,
        "transcription_seconds": round(sum(r.get("seconds", 0.0) for r in rows), 2),
    }
    audio_total = sum(r.get("audio_seconds", 0.0) for r in rows)
    if audio_total:
        summary["audio_seconds"] = round(audio_total, 2)
        summary["rtf"] = round(summary["transcription_seconds"] / audio_total, 4)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = summary["scores"]
    print(
        f"{args.system} on {label}: {s['utterances']} utt, "
        f"WER {s['wer']:.2%}, CER {s['cer']:.2%}, diacritics {s['diacritics_error_rate']:.2%} -> {summary_path}"
    )
    return 0


def _pct(x: Optional[float]) -> str:
    return "–" if x is None else f"{x * 100:.2f} %"


def cmd_report(args: argparse.Namespace) -> int:
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.results).glob("*.summary.json"))]
    if not summaries:
        print(f"no *.summary.json files in {args.results}", file=sys.stderr)
        return 1
    summaries.sort(key=lambda s: (s["dataset"], s["scores"]["wer"]))
    if args.format == "json":
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0
    print("| Dataset | System | Utterances | WER | CER | Diacritics errors | Numbers | Names | RTF |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        sc, ent = s["scores"], s.get("entities", {})
        print(
            f"| {s['dataset']} | {s['system']} | {sc['utterances']} | {_pct(sc['wer'])} | {_pct(sc['cer'])} | "
            f"{_pct(sc['diacritics_error_rate'])} | {_pct(ent.get('num', {}).get('accuracy'))} | "
            f"{_pct(ent.get('name', {}).get('accuracy'))} | {s['rtf'] if 'rtf' in s else '–'} |"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sk-asr-bench", description="Benchmark Slovak speech recognition.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="transcribe a dataset with one system and score it")
    r.add_argument("--system", required=True, help="hyp:PATH | faster-whisper:MODEL | transformers:MODEL | openai:MODEL")
    r.add_argument("--dataset", required=True, help="jsonl:PATH | fleurs | commonvoice:DIR")
    r.add_argument("--limit", type=int, default=None, help="only the first N utterances")
    r.add_argument("--out", default="results", help="output directory (default: results)")
    r.add_argument("--batch-size", type=int, default=1, help="utterances per model call, for systems that support batching (default: 1)")
    r.add_argument("--fresh", action="store_true", help="ignore cached transcripts and start over")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="print a leaderboard from result summaries")
    rep.add_argument("--results", default="results")
    rep.add_argument("--format", choices=["md", "json"], default="md")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
