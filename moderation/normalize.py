"""Нормализация текста против обфускации: «х у й», «xуй», «cyka», «3ло», «тупоооой», zero-width символы.

Нормализованный текст используется только для детекции (правила и модели), исходный текст не меняется.
"""
import re
import unicodedata

# Латиница / цифры, визуально похожие на кириллицу
_LOOKALIKE = "aceopxykmthb0@346"
_CYRILLIC = "асеорхукмтнвоазчб"
_TO_CYR = str.maketrans(_LOOKALIKE, _CYRILLIC)

_ZERO_WIDTH = re.compile(r"[​-‏⁠-⁤﻿­͏]")
_CYR_CHAR = re.compile(r"[а-я]")
_LAT_CHAR = re.compile(r"[a-z]")
_LOOKALIKE_TOKEN = re.compile(rf"^[{_LOOKALIKE}]+$")
_TOKEN = re.compile(r"[\w@]+")
# Разрядка: 3+ одиночных символа через пробел / точку / дефис / подчёркивание («х у й», «х.у.й»)
_SPACED = re.compile(r"(?<![\w@])(?:[\w@][\s.\-_]+){2,}[\w@](?![\w@])")
_SEPARATORS = re.compile(r"[\s.\-_]+")
_REPEATS = re.compile(r"([^\W\d_])\1{2,}")
_WHITESPACE = re.compile(r"\s+")


def _map_token(token: str, cyrillic_text: bool) -> str:
    if not any(ch.isalpha() for ch in token):
        return token  # числа («2026») не трогаем
    if _CYR_CHAR.search(token) or (cyrillic_text and _LOOKALIKE_TOKEN.match(token)):
        return token.translate(_TO_CYR)
    return token  # обычные латинские слова («iphone», «covid19») не трогаем


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = _ZERO_WIDTH.sub("", t).lower().replace("ё", "е")
    t = _SPACED.sub(lambda m: _SEPARATORS.sub("", m.group()), t)
    # «русский текст»: кириллических слов не меньше, чем обычных латинских (не состоящих из двойников)
    tokens = _TOKEN.findall(t)
    n_cyr = sum(bool(_CYR_CHAR.search(w)) for w in tokens)
    n_lat = sum(bool(_LAT_CHAR.search(w)) and not _LOOKALIKE_TOKEN.match(w) for w in tokens)
    cyrillic_text = n_cyr > 0 and n_cyr >= n_lat
    t = _TOKEN.sub(lambda m: _map_token(m.group(), cyrillic_text), t)
    t = _REPEATS.sub(r"\1", t)
    return _WHITESPACE.sub(" ", t).strip()
