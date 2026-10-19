"""Правила и словари: категории нарушений и признаки для ансамбля."""
import re
from dataclasses import dataclass

from moderation import lexicon as lx

CATEGORIES = ("profanity", "insult", "threat", "illegal_activity")
GENERIC_CATEGORY = "toxicity"  # нарушение найдено моделями, но конкретная категория не определена


@dataclass(frozen=True)
class RuleHit:
    category: str
    strong: bool
    span: str


def _compile(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns))


_MASK_TOKEN = re.compile(r"[а-яa-z]*[*#%]+[а-яa-z*#%]*")


class RuleEngine:
    def __init__(self):
        self._groups = [
            ("profanity", True, _compile(lx.PROFANITY_STRONG)),
            ("insult", True, _compile(lx.INSULT_STRONG)),
            ("insult", False, _compile(lx.INSULT_WEAK)),
            ("threat", True, _compile(lx.THREAT_STRONG)),
            ("threat", False, _compile(lx.THREAT_WEAK)),
            ("illegal_activity", True, _compile(lx.ILLEGAL_STRONG)),
            ("illegal_activity", False, _compile(lx.ILLEGAL_WEAK)),
        ]
        self._drugs = _compile(lx.DRUGS)
        self._commerce = _compile(lx.COMMERCE)

    def match(self, text: str) -> list[RuleHit]:
        """text — нормализованный текст."""
        hits = []
        for category, strong, pattern in self._groups:
            hits += [RuleHit(category, strong, m.group().strip()) for m in pattern.finditer(text)]

        drugs = [m.group() for m in self._drugs.finditer(text)]
        if drugs:  # упоминание наркотиков + признак торговли = сбыт → strong
            strong = bool(self._commerce.search(text))
            hits += [RuleHit("illegal_activity", strong, d) for d in drugs]

        hits += self._masked(text)
        return hits

    @staticmethod
    def _masked(text: str) -> list[RuleHit]:
        hits = []
        for m in _MASK_TOKEN.finditer(text):
            token = m.group()
            if sum(ch.isalpha() for ch in token) < 2:
                continue
            regex = re.compile("".join("[а-я]{1,3}" if ch in "*#%" else re.escape(ch)
                                       for ch in re.sub(r"[*#%]+", "*", token)))
            for (category, strong), words in lx.MASK_TARGETS.items():
                if any(regex.fullmatch(w) for w in words):
                    hits.append(RuleHit(category, strong, token))
                    break
        return hits


def categories_of(hits: list[RuleHit]) -> list[str]:
    found = {h.category for h in hits}
    return [c for c in CATEGORIES if c in found]
