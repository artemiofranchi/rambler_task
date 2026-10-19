import pytest

from moderation.normalize import normalize
from moderation.rules import RuleEngine, categories_of

engine = RuleEngine()


def hits(text):
    return engine.match(normalize(text))


@pytest.mark.parametrize("text, category", [
    ("да пошёл ты нахуй", "profanity"),
    ("х у й тебе", "profanity"),
    ("ну ты и долбоёб", "profanity"),
    ("б**дь, опять", "profanity"),
    ("blyat", "profanity"),
    ("я тебя найду", "threat"),
    ("убью тебя", "threat"),
    ("знаю где ты живешь", "threat"),
    ("голову оторву", "threat"),
    ("продам права без экзамена", "illegal_activity"),
    ("закладки меф, пиши в телегу", "illegal_activity"),
    ("продам пистолет", "illegal_activity"),
])
def test_strong_violations(text, category):
    h = hits(text)
    assert any(x.strong and x.category == category for x in h), h


@pytest.mark.parametrize("text, category", [
    ("ты дебил", "insult"),
    ("тупой нож", "insult"),               # слабый сигнал: решение примет модель
    ("сдохни", "threat"),
    ("героин нашли при обыске", "illegal_activity"),
    ("де*ил", "insult"),
])
def test_weak_signals(text, category):
    h = hits(text)
    assert h and all(not x.strong for x in h) and category in categories_of(h), h


@pytest.mark.parametrize("text", [
    "Небо голубое, хлеб свежий", "требуется ребенок", "бледный хулиган", "Забью гол в субботу",
    "Ебола в Африке", "Педро Альмодовар", "жидкий азот", "хачапури", "мандарин", "сукно и сук дерева",
    "Хуан Карлос", "купил хлеб и масло", "оскорбление чувств", "цена 5*3",
])
def test_no_false_positives(text):
    assert hits(text) == []
