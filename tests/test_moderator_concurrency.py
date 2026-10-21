"""Регрессия: параллельные вызовы из микробатчера и /moderate/batch не должны входить в модель одновременно
(на Apple MPS это роняло процесс: failed assertion in IOGPUMetalCommandBuffer)."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from moderation.moderator import Moderator
from moderation.rules import RuleEngine


class ReentrancyDetector:
    def __init__(self):
        self.active, self.overlaps = 0, 0
        self.guard = threading.Lock()

    def predict_proba(self, texts):
        with self.guard:
            self.active += 1
            self.overlaps += self.active > 1
        time.sleep(0.01)
        with self.guard:
            self.active -= 1
        return np.zeros(len(texts))


def test_model_calls_are_serialized():
    mod = Moderator.__new__(Moderator)
    mod.use_normalization, mod.max_chars, mod.rules = True, 5000, RuleEngine()
    detector = ReentrancyDetector()
    mod.models, mod._lock = {"fake": detector}, threading.Lock()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda _: mod.signals(["текст"]), range(32)))
    assert detector.overlaps == 0
