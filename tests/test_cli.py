import io
import json
import struct
import wave
from pathlib import Path

from sk_asr_bench.cli import main
from sk_asr_bench.systems import wav_bytes

FIX = Path(__file__).parent / "fixtures"


def test_run_and_report(tmp_path, capsys):
    out = tmp_path / "res"
    rc = main(["run", "--system", f"hyp:{FIX/'hyp.jsonl'}", "--dataset", f"jsonl:{FIX/'refs.jsonl'}", "--out", str(out)])
    assert rc == 0
    summary = json.loads(next(out.glob("*.summary.json")).read_text(encoding="utf-8"))
    sc = summary["scores"]
    assert sc["utterances"] == 3
    # u1: "novák"->"nóvak" and u3: "učiteľ"->"ucitel" are both diacritics-only substitutions; 17 reference words
    assert sc["ref_words"] == 17
    assert abs(sc["wer"] - 2 / 17) < 1e-9
    assert abs(sc["diacritics_error_rate"] - 2 / 17) < 1e-9
    assert summary["entities"]["name"]["accuracy"] == 0.0
    assert summary["entities"]["num"]["accuracy"] == 1.0

    # second run resumes from cache without needing the system
    assert main(["run", "--system", f"hyp:{FIX/'hyp.jsonl'}", "--dataset", f"jsonl:{FIX/'refs.jsonl'}", "--out", str(out)]) == 0

    capsys.readouterr()
    assert main(["report", "--results", str(out)]) == 0
    table = capsys.readouterr().out
    assert "| Dataset | System |" in table and "11.76 %" in table


def test_limit(tmp_path):
    out = tmp_path / "res"
    main(["run", "--system", f"hyp:{FIX/'hyp.jsonl'}", "--dataset", f"jsonl:{FIX/'refs.jsonl'}", "--out", str(out), "--limit", "1"])
    summary = json.loads(next(out.glob("*.summary.json")).read_text(encoding="utf-8"))
    assert summary["scores"]["utterances"] == 1


def test_wav_bytes_roundtrip():
    data = wav_bytes([0.0, 0.5, -1.0, 2.0], 16000)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        frames = struct.unpack("<4h", w.readframes(4))
    assert frames == (0, 16383, -32767, 32767)
