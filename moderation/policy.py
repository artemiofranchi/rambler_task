"""Ансамбль и политика решений.

1. Стекинг: логистическая регрессия поверх сигналов трёх алгоритмов
   (логит линейной модели, логит трансформера, strong/weak-срабатывания правил) → калиброванная P(нарушение).
2. Пороги:  P < t_allow → ALLOW;  P ≥ t_block → BLOCK;  между ними → REVIEW;  strong-правило → BLOCK.
   t_allow — максимальный порог, при котором recall нарушений (BLOCK + REVIEW) ≥ target_recall:
             пропуск нарушения критичнее, чем лишняя проверка.
   t_block — минимальный порог ≥ t_allow, при котором точность BLOCK ≥ target_block_precision:
             так зона REVIEW получается минимальной при заданном качестве автоблокировки.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ALLOW, REVIEW, BLOCK = "ALLOW", "REVIEW", "BLOCK"
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def build_features(model_probs: dict, strong: np.ndarray, weak: np.ndarray, names: list[str]) -> np.ndarray:
    cols = []
    for name in names:
        if name == "rule_strong":
            cols.append(np.asarray(strong, dtype=np.float64))
        elif name == "rule_weak":
            cols.append(np.asarray(weak, dtype=np.float64))
        else:
            cols.append(_logit(model_probs[name]))
    return np.column_stack(cols)


@dataclass
class Policy:
    feature_names: list[str]
    coef: list[float]
    intercept: float
    t_allow: float
    t_block: float
    strong_rule_confidence: float = 0.99
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- инференс
    def score(self, model_probs: dict, strong, weak) -> np.ndarray:
        X = build_features(model_probs, strong, weak, self.feature_names)
        return 1 / (1 + np.exp(-(X @ np.asarray(self.coef) + self.intercept)))

    def decide(self, score: np.ndarray, strong: np.ndarray) -> list[str]:
        return [BLOCK if s_hit or p >= self.t_block else (ALLOW if p < self.t_allow else REVIEW)
                for p, s_hit in zip(score, strong)]

    # ---------------------------------------------------------------- обучение
    @classmethod
    def fit(cls, model_probs_a, strong_a, weak_a, y_a, model_probs_b, strong_b, weak_b, y_b,
            feature_names, target_recall, target_block_precision, strong_rule_confidence=0.99):
        """Часть A валидации — веса стекинга, часть B — пороги (чтобы не подбирать всё на одних данных).

        Стекинг учится только на текстах без strong-срабатываний: они и так получают BLOCK,
        а модели нужны именно для неочевидных случаев."""
        rest = ~np.asarray(strong_a, bool)
        stacker = LogisticRegression(C=1.0, max_iter=1000)
        stacker.fit(build_features({k: np.asarray(v)[rest] for k, v in model_probs_a.items()},
                                   np.asarray(strong_a)[rest], np.asarray(weak_a)[rest], feature_names),
                    np.asarray(y_a)[rest])
        policy = cls(feature_names, stacker.coef_[0].tolist(), float(stacker.intercept_[0]), 0.5, 0.5,
                     strong_rule_confidence)
        score_b = policy.score(model_probs_b, strong_b, weak_b)
        policy.t_allow, policy.t_block, info = choose_thresholds(
            score_b, np.asarray(strong_b, bool), np.asarray(y_b), target_recall, target_block_precision)
        policy.meta = {"target_recall": target_recall, "target_block_precision": target_block_precision,
                       "calibration_set": info}
        return policy

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Policy":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def decision_stats(decisions, y) -> dict:
    d, y = np.asarray(decisions), np.asarray(y).astype(bool)
    block, allow = d == BLOCK, d == ALLOW
    n_pos = max(y.sum(), 1)
    return {
        "share_allow": float(allow.mean()), "share_review": float((d == REVIEW).mean()),
        "share_block": float(block.mean()),
        "recall_violations": float((~allow & y).sum() / n_pos),     # нарушение не получило ALLOW
        "miss_rate": float((allow & y).sum() / n_pos),              # нарушение опубликовано
        "auto_recall": float((block & y).sum() / n_pos),            # нарушение заблокировано автоматически
        "block_precision": float((block & y).sum() / max(block.sum(), 1)),
        "false_block_rate": float((block & ~y).sum() / max((~y).sum(), 1)),  # нормальное заблокировано
    }


def choose_thresholds(score, strong, y, target_recall, target_block_precision):
    y = y.astype(bool)
    n_pos = max(y.sum(), 1)
    candidates = np.unique(np.concatenate([[0.0, 1.0 + _EPS], score]))

    # t_allow: максимальный порог, при котором recall(BLOCK+REVIEW) >= target
    t_allow = 0.0
    for t in candidates:
        if ((strong | (score >= t)) & y).sum() / n_pos >= target_recall:
            t_allow = t
        else:
            break

    # t_block: минимальный порог >= t_allow, при котором precision(BLOCK) >= target
    t_block = 1.0 + _EPS
    for t in candidates[candidates >= t_allow]:
        block = strong | (score >= t)
        if block.sum() and (block & y).sum() / block.sum() >= target_block_precision:
            t_block = t
            break

    decisions = [BLOCK if s or p >= t_block else (ALLOW if p < t_allow else REVIEW) for p, s in zip(score, strong)]
    return float(t_allow), float(min(t_block, 1.0 + _EPS)), decision_stats(decisions, y)
