"""Moderator: нормализация → правила + линейная модель + трансформер → стекинг → решение."""
import threading
from dataclasses import dataclass, field

import numpy as np

from moderation.config import load_config, resolve
from moderation.models.linear import LinearModel
from moderation.normalize import normalize
from moderation.policy import ALLOW, BLOCK, Policy
from moderation.rules import GENERIC_CATEGORY, RuleEngine, categories_of


@dataclass
class ModerationResult:
    decision: str
    categories: list[str]
    confidence: float
    evidence: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)


@dataclass
class Signals:
    normalized: list[str]
    hits: list[list]
    strong: np.ndarray
    weak: np.ndarray
    probs: dict  # имя модели → P(нарушение)


class Moderator:
    def __init__(self, cfg: dict | None = None, *, use_normalization: bool | None = None, load_policy: bool = True):
        self.cfg = cfg or load_config()
        self.use_normalization = (self.cfg["normalization"]["enabled"]
                                  if use_normalization is None else use_normalization)
        self.max_chars = self.cfg["service"]["max_text_chars"]
        self.rules = RuleEngine()
        self._lock = threading.Lock()
        self.models = {"linear": LinearModel.load(resolve(self.cfg["linear"]["path"]))}
        if self.cfg["transformer"]["enabled"]:
            from moderation.models.transformer import TransformerModel  # torch грузим только если нужен
            self.models["transformer"] = TransformerModel.from_config(
                self.cfg["transformer"], resolve(self.cfg["transformer"]["path"]))
        self.policy = Policy.load(resolve(self.cfg["policy"]["path"])) if load_policy else None
        if self.policy:
            missing = {n for n in self.policy.feature_names if not n.startswith("rule_")} - set(self.models)
            if missing:
                raise RuntimeError(f"policy.json обучен с моделями {sorted(missing)}, но они выключены в конфиге. "
                                   f"Включите их или перекалибруйте: make calibrate")

    @property
    def components(self) -> list[str]:
        return ["rules", *self.models]

    def signals(self, texts: list[str]) -> Signals:
        texts = [(t or "")[: self.max_chars] for t in texts]
        norm = [normalize(t) if self.use_normalization else (t or "").lower() for t in texts]
        hits = [self.rules.match(t) for t in norm]
        strong = np.array([any(h.strong for h in hs) for hs in hits], dtype=bool)
        weak = np.array([any(not h.strong for h in hs) for hs in hits], dtype=bool)
        with self._lock:
            probs = {name: m.predict_proba(norm) for name, m in self.models.items()}
        return Signals(norm, hits, strong, weak, probs)

    def moderate_batch(self, texts: list[str]) -> list[ModerationResult]:
        if not texts:
            return []
        sig = self.signals(texts)
        score = self.policy.score(sig.probs, sig.strong, sig.weak)
        decisions = self.policy.decide(score, sig.strong)
        results = []
        for i, decision in enumerate(decisions):
            if not sig.normalized[i]:
                results.append(ModerationResult(ALLOW, [], 1.0))
                continue
            p = float(score[i])
            if decision == BLOCK:
                conf = max(p, self.policy.strong_rule_confidence) if sig.strong[i] else p
            elif decision == ALLOW:
                conf = 1.0 - p
            else:
                conf = p
            cats = [] if decision == ALLOW else (categories_of(sig.hits[i]) or [GENERIC_CATEGORY])
            evidence = list(dict.fromkeys(h.span for h in sig.hits[i]))[:10]
            scores = {name: round(float(v[i]), 4) for name, v in sig.probs.items()}
            scores["ensemble"] = round(p, 4)
            results.append(ModerationResult(decision, cats, round(conf, 4), evidence, scores))
        return results

    def moderate(self, text: str) -> ModerationResult:
        return self.moderate_batch([text])[0]
