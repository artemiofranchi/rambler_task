"""Искажения текста для проверки устойчивости — те же приёмы, которыми пользователи обходят модерацию."""
import random
import re

_CYR2LAT = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "к": "k", "м": "m",
            "т": "t", "н": "h", "в": "b"}
_LEET = {"о": "0", "з": "3", "ч": "4", "б": "6", "а": "@"}
_VOWELS = set("аеиоуыэюя")


def homoglyph(word: str, rng: random.Random) -> str:
    return "".join(_CYR2LAT[c] if c in _CYR2LAT and rng.random() < 0.6 else c for c in word)


def leet(word: str, rng: random.Random) -> str:
    return "".join(_LEET[c] if c in _LEET and rng.random() < 0.7 else c for c in word)


def spacing(word: str, rng: random.Random) -> str:
    return rng.choice([" ", ".", "-"]).join(word)


def repeat(word: str, rng: random.Random) -> str:
    idx = [i for i, c in enumerate(word) if c in _VOWELS] or [len(word) - 1]
    i = rng.choice(idx)
    return word[:i] + word[i] * rng.randint(3, 5) + word[i + 1:]


def mask(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + "*" * rng.randint(1, 2) + word[i + 1:]


def zero_width(word: str, rng: random.Random) -> str:
    return "​".join(word)


PERTURBATIONS = {f.__name__: f for f in (homoglyph, leet, spacing, repeat, mask, zero_width)}


def perturb_text(text: str, kind: str, rng: random.Random, targets=(), rate: float = 0.3) -> str:
    """Искажает слова-мишени (например, найденные правилами) и случайную долю rate остальных слов."""
    fn = PERTURBATIONS[kind]
    targets = {t.lower().replace("ё", "е") for t in targets}

    def repl(m):
        low = m.group().lower()
        hit = low.replace("ё", "е") in targets or rng.random() < rate
        return fn(low, rng) if hit else m.group()

    return _WORD.sub(repl, text)


_WORD = re.compile(r"[а-яёА-ЯЁ]{3,}")
