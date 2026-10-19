"""Сквозной тест на обученных артефактах. Пропускается, если модели ещё не обучены / не скачаны."""
import json

import pytest

from moderation.config import load_config, resolve

cfg = load_config()
artifacts = [resolve(cfg["linear"]["path"]), resolve(cfg["policy"]["path"])]
if cfg["transformer"]["enabled"]:
    artifacts.append(resolve(cfg["transformer"]["path"]))
pytestmark = pytest.mark.skipif(not all(p.exists() for p in artifacts), reason="модели не найдены: make train")


@pytest.fixture(scope="module")
def moderator():
    from moderation.moderator import Moderator
    return Moderator(cfg)


def test_samples(moderator):
    samples = [json.loads(line) for line in
               resolve("data/samples/messages.jsonl").read_text(encoding="utf-8").splitlines() if line]
    results = moderator.moderate_batch([s["text"] for s in samples])
    for s, r in zip(samples, results):
        assert r.decision in ("ALLOW", "REVIEW", "BLOCK")
        assert 0 <= r.confidence <= 1
        assert (r.decision == "ALLOW") == (r.categories == [])  # у нарушения всегда есть причина
        if s["expected"] == "BLOCK" and any(w in s["text"].lower() for w in ("нахуй", "x y й", "урою", "закладки", "права")):
            assert r.decision == "BLOCK", (s["text"], r)  # однозначные правила блокируют всегда


def test_empty_text(moderator):
    assert moderator.moderate("   ").decision == "ALLOW"
