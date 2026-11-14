import pytest

from sk_asr_bench import align, cer, diacritics_error_rate, score_corpus, wer


def test_perfect():
    assert wer("Dobrý deň.", "dobrý deň") == 0.0
    assert cer("Dobrý deň.", "dobrý deň") == 0.0


def test_one_substitution():
    assert wer("a b c d", "a x c d") == pytest.approx(0.25)


def test_deletion_and_insertion():
    a = align("a b c d".split(), "a c d".split())
    assert (a.substitutions, a.deletions, a.insertions) == (0, 1, 0)
    assert a.errors == 1 and a.ref_len == 4
    a = align("a c".split(), "a b c".split())
    assert (a.substitutions, a.deletions, a.insertions) == (0, 0, 1)


def test_empty_reference():
    assert wer("", "") == 0.0
    assert wer("", "niečo") == 1.0


def test_wer_can_exceed_one():
    assert wer("a", "x y z") == pytest.approx(3.0)


def test_cer():
    assert cer("mačka", "macka") == pytest.approx(1 / 5)


def test_diacritics_error_rate():
    # "ucitel" vs "učiteľ" is a diacritics-only error; "kniha" vs "knihu" is not
    assert diacritics_error_rate("učiteľ číta knihu", "ucitel číta kniha") == pytest.approx(1 / 3)


def test_corpus_is_pooled_not_averaged():
    s = score_corpus([("a", "x"), ("a b c d e f g h i j", "a b c d e f g h i j")])
    assert s.wer == pytest.approx(1 / 11)
    assert s.utterances == 2 and s.ref_words == 11
