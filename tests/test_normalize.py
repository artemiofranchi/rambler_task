import pytest

from moderation.normalize import normalize


@pytest.mark.parametrize("raw, expected", [
    ("х у й", "хуй"),                       # разрядка
    ("х.у.й", "хуй"),
    ("xуй", "хуй"),                         # латиница внутри кириллического слова
    ("Ты cyka", "ты сука"),                 # полностью латинское слово в русском тексте
    ("3л0й", "злой"),                       # цифры вместо букв
    ("тупоооой", "тупой"),                  # повторы
    ("ху​й", "хуй"),                   # невидимые символы
    ("ЕЩЁ   раз", "еще раз"),               # регистр, ё, пробелы
])
def test_deobfuscation(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("text", ["iphone 15 pro в 2026", "covid19", "привет, hello world"])
def test_keeps_regular_latin_and_numbers(text):
    assert normalize(text) == text


def test_does_not_join_regular_short_words():
    assert normalize("я и ты") == "я и ты"


def test_empty():
    assert normalize("") == "" and normalize(None) == ""
