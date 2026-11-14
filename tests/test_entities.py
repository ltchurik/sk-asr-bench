from sk_asr_bench import Entity, entity_accuracy, parse_tagged


def test_parse_tagged():
    plain, ents = parse_tagged("Platba {num:1 250} eur od {name:Jána Nováka}.")
    assert plain == "Platba 1 250 eur od Jána Nováka."
    assert ents == [Entity("num", "1 250"), Entity("name", "Jána Nováka")]


def test_entity_accuracy():
    pairs = [
        ("Volám sa {name:Ján Novák}.", "volám sa ján novák"),
        ("Volám sa {name:Ján Novák}.", "volám sa ján nóvak"),
        ("Zaplatil {num:1 250} eur.", "zaplatil 1250 eur"),
    ]
    acc = entity_accuracy(pairs)
    assert acc["name"] == {"total": 2, "correct": 1, "accuracy": 0.5}
    assert acc["num"]["accuracy"] == 1.0
