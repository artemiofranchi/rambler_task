import numpy as np

from moderation.policy import ALLOW, BLOCK, REVIEW, Policy, choose_thresholds, decision_stats


def test_thresholds_meet_targets():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.3
    score = np.clip(np.where(y, rng.normal(0.75, 0.15, 5000), rng.normal(0.25, 0.15, 5000)), 0, 1)
    strong = np.zeros(5000, bool)
    t_allow, t_block, st = choose_thresholds(score, strong, y, target_recall=0.95, target_block_precision=0.9)
    assert t_allow <= t_block
    assert st["recall_violations"] >= 0.95
    assert st["block_precision"] >= 0.9
    assert abs(st["share_allow"] + st["share_review"] + st["share_block"] - 1) < 1e-9


def test_strong_rule_always_blocks():
    policy = Policy(["linear"], [1.0], 0.0, t_allow=0.3, t_block=0.8)
    decisions = policy.decide(np.array([0.01, 0.5, 0.9, 0.01]), np.array([False, False, False, True]))
    assert decisions == [ALLOW, REVIEW, BLOCK, BLOCK]


def test_save_load_roundtrip(tmp_path):
    policy = Policy(["linear", "rule_weak"], [1.5, 0.3], -0.2, 0.3, 0.8, meta={"x": 1})
    policy.save(tmp_path / "p.json")
    assert Policy.load(tmp_path / "p.json") == policy


def test_decision_stats():
    st = decision_stats([ALLOW, BLOCK, REVIEW, BLOCK], [1, 1, 1, 0])
    assert st["miss_rate"] == 1 / 3 and st["block_precision"] == 0.5 and st["false_block_rate"] == 1.0
