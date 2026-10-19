"""Калибровка ансамбля на валидации: веса стекинга (часть A) + пороги ALLOW/REVIEW/BLOCK (часть B).

Пишет models/policy.json и reports/tradeoff.md — как меняются доля REVIEW и ошибки при других целевых значениях.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.config import load_config, resolve  # noqa: E402
from moderation.data import load_split  # noqa: E402
from moderation.moderator import Moderator  # noqa: E402
from moderation.policy import Policy, choose_thresholds  # noqa: E402



def main():
    cfg = load_config()
    REPORTS = resolve(cfg.get("reports_dir", "reports"))
    pcfg = cfg["policy"]
    mod = Moderator(cfg, load_policy=False)
    val = load_split(cfg, "val")
    y = val["label"].to_numpy()
    print(f"сигналы компонентов на валидации ({len(val)} текстов)…")
    sig = mod.signals(val["text"].tolist())

    idx_a, idx_b = train_test_split(np.arange(len(val)), test_size=0.5, stratify=y, random_state=cfg["data"]["seed"])
    part = lambda idx: ({k: v[idx] for k, v in sig.probs.items()}, sig.strong[idx], sig.weak[idx], y[idx])
    names = [*mod.models, "rule_weak"]

    policy = Policy.fit(*part(idx_a), *part(idx_b), feature_names=names,
                        target_recall=pcfg["target_recall"], target_block_precision=pcfg["target_block_precision"],
                        strong_rule_confidence=pcfg["strong_rule_confidence"])
    policy.save(resolve(pcfg["path"]))
    print(json.dumps({k: v for k, v in policy.__dict__.items()}, ensure_ascii=False, indent=2))

    # компромиссы: как меняется доля REVIEW при других целях
    probs_b, strong_b, weak_b, y_b = part(idx_b)
    score_b = policy.score(probs_b, strong_b, weak_b)
    rows = []
    for r in (0.90, 0.93, 0.95, 0.97, 0.99):
        for p in (0.85, 0.90, 0.95):
            t_allow, t_block, st = choose_thresholds(score_b, strong_b, y_b, r, p)
            rows.append({"target_recall": r, "target_block_precision": p, "t_allow": round(t_allow, 4),
                         "t_block": round(t_block, 4), **{k: round(v, 4) for k, v in st.items()}})
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "tradeoff.json").write_text(json.dumps(rows, indent=2))
    lines = ["| target recall | target BLOCK precision | ALLOW | REVIEW | BLOCK | recall | miss rate | "
             "BLOCK precision | false block |", "|" + "---|" * 9]
    for row in rows:
        lines.append(f"| {row['target_recall']} | {row['target_block_precision']} | {row['share_allow']:.1%} | "
                     f"{row['share_review']:.1%} | {row['share_block']:.1%} | {row['recall_violations']:.1%} | "
                     f"{row['miss_rate']:.1%} | {row['block_precision']:.1%} | {row['false_block_rate']:.1%} |")
    (REPORTS / "tradeoff.md").write_text("# Компромиссы порогов (валидация, часть B)\n\n" + "\n".join(lines) + "\n")
    print(f"\nсохранено: {resolve(pcfg['path'])}, {REPORTS / 'tradeoff.md'}")


if __name__ == "__main__":
    main()
