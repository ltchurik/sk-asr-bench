import unicodedata

from sk_asr_bench import normalize, strip_diacritics


def test_case_punctuation_whitespace():
    assert normalize("  Dobrý   deň, svet!  ") == "dobrý deň svet"


def test_quotes_and_dashes_removed():
    assert normalize("„Áno“ – povedal…") == "áno povedal"


def test_diacritics_are_kept():
    assert normalize("Ľščťžýáíéďňôúäŕĺ") == "ľščťžýáíéďňôúäŕĺ"


def test_nfc():
    decomposed = unicodedata.normalize("NFD", "kôň")
    assert normalize(decomposed) == "kôň"


def test_numbers_keep_internal_punctuation():
    assert normalize("Stálo to 3,5 eura o 10:30.") == "stálo to 3,5 eura o 10:30"


def test_thousands_separators_joined():
    assert normalize("1 250 eur") == "1250 eur"
    assert normalize("1 250 000 eur") == "1250000 eur"


def test_number_hook():
    assert normalize("mám 2 psy", number_hook=lambda t: t.replace("2", "dva")) == "mám dva psy"


def test_strip_diacritics():
    assert strip_diacritics("Učiteľ v kúpeľni") == "Ucitel v kupelni"
