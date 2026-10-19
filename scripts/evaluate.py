"""Оценка качества на тестовой выборке → reports/metrics.json, reports/metrics.md.

1. Компоненты по отдельности (правила, линейная модель, трансформер) и ансамбль: ROC-AUC, PR-AUC, F1.
2. Решения сервиса ALLOW/REVIEW/BLOCK: доли, пропуски нарушений, ложные блокировки.
3. Устойчивость: те же метрики на искажённых текстах (латиница, leet, разрядка, повторы, звёздочки,
   невидимые символы) — с нормализацией и без неё.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.config import load_config, resolve  # noqa: E402
from moderation.data import Timer, load_split, stratified_sample  # noqa: E402
from moderation.moderator import Moderator  # noqa: E402
from moderation.perturb import PERTURBATIONS, perturb_text  # noqa: E402
from moderation.policy import decision_stats  # noqa: E402
from moderation.rules import categories_of  # noqa: E402


def binary_metrics(y, p, threshold=0.5):
    pred = p >= threshold
    out = {"precision@0.5": precision_score(y, pred, zero_division=0), "recall@0.5": recall_score(y, pred),
           "f1@0.5": f1_score(y, pred)}
    if len(np.unique(p)) > 2:
        out.update({"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p)})
    return {k: round(float(v), 4) for k, v in out.items()}


def run(mod, texts, y):
    sig = mod.signals(texts)
    score = mod.policy.score(sig.probs, sig.strong, sig.weak)
    decisions = mod.policy.decide(score, sig.strong)
    stats = {k: round(v, 4) for k, v in decision_stats(decisions, y).items()}
    stats["ensemble_roc_auc"] = round(float(roc_auc_score(y, score)), 4)
    return sig, score, decisions, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None, help="ограничить тест (по умолчанию весь)")
    ap.add_argument("--robustness-samples", type=int, default=3000)
    args = ap.parse_args()
    cfg = load_config()
    REPORTS = resolve(cfg.get("reports_dir", "reports"))
    seed = cfg["data"]["seed"]
    mod = Moderator(cfg)

    test = stratified_sample(load_split(cfg, "test"), args.max_samples, seed)
    texts, y = test["text"].tolist(), test["label"].to_numpy()
    print(f"тест: {len(test)} текстов, доля токсичных {y.mean():.3f}")

    with Timer() as t:
        sig, score, decisions, stats = run(mod, texts, y)
    report = {"test_size": len(test), "toxic_share": round(float(y.mean()), 4),
              "throughput_offline_msgs_per_min": round(len(texts) / t.seconds * 60),
              "components": {name: binary_metrics(y, p) for name, p in sig.probs.items()},
              "decisions": stats, "policy": {"t_allow": mod.policy.t_allow, "t_block": mod.policy.t_block,
                                             **mod.policy.meta}}
    report["components"]["rules_strong"] = binary_metrics(y, sig.strong.astype(float))
    report["components"]["ensemble"] = binary_metrics(y, score)

    flagged = [i for i, d in enumerate(decisions) if d != "ALLOW"]
    cats = Counter(c for i in flagged for c in (categories_of(sig.hits[i]) or ["toxicity"]))
    report["categories_in_flagged"] = {k: round(v / max(len(flagged), 1), 4) for k, v in cats.most_common()}
    report["examples"] = {d: [texts[i][:150] for i in np.where(np.asarray(decisions) == d)[0][:3]]
                          for d in ("ALLOW", "REVIEW", "BLOCK")}

    # ------------------------------------------------------------ устойчивость
    rob = stratified_sample(test, args.robustness_samples, seed)
    r_texts, r_y = rob["text"].tolist(), rob["label"].to_numpy()
    clean_hits = mod.signals(r_texts).hits
    robustness = {}
    for kind in ["clean", *PERTURBATIONS]:
        rng = random.Random(seed)
        variant = r_texts if kind == "clean" else [
            perturb_text(t, kind, rng, targets=[h.span for h in hs]) for t, hs in zip(r_texts, clean_hits)]
        row = {}
        for norm_on in (True, False):
            mod.use_normalization = norm_on
            _, _, _, st = run(mod, variant, r_y)
            row["with_normalization" if norm_on else "without_normalization"] = st
        mod.use_normalization = cfg["normalization"]["enabled"]
        robustness[kind] = row
        print(f"устойчивость {kind:12} recall: {row['with_normalization']['recall_violations']:.3f} "
              f"(без нормализации {row['without_normalization']['recall_violations']:.3f})")
    report["robustness"] = {"samples": len(rob), "results": robustness}

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (REPORTS / "metrics.md").write_text(to_markdown(report))
    print((REPORTS / "metrics.md").read_text())


def to_markdown(r):
    pct = lambda v: f"{v:.1%}"
    lines = [f"# Метрики качества (тест: {r['test_size']} текстов, токсичных {pct(r['toxic_share'])})", "",
             "## Компоненты", "", "| компонент | ROC-AUC | PR-AUC | precision@0.5 | recall@0.5 | F1@0.5 |",
             "|---|---|---|---|---|---|"]
    for name, m in r["components"].items():
        lines.append(f"| {name} | {m.get('roc_auc', '—')} | {m.get('pr_auc', '—')} | {m['precision@0.5']} | "
                     f"{m['recall@0.5']} | {m['f1@0.5']} |")
    d = r["decisions"]
    lines += ["", "## Решения сервиса", "",
              f"Пороги: ALLOW < {r['policy']['t_allow']:.3f} ≤ REVIEW < {r['policy']['t_block']:.3f} ≤ BLOCK", "",
              "| ALLOW | REVIEW | BLOCK | нарушения пойманы (BLOCK+REVIEW) | пропущено нарушений | "
              "заблокировано автоматически | точность BLOCK | ложные блокировки |",
              "|---|---|---|---|---|---|---|---|",
              f"| {pct(d['share_allow'])} | {pct(d['share_review'])} | {pct(d['share_block'])} | "
              f"{pct(d['recall_violations'])} | {pct(d['miss_rate'])} | {pct(d['auto_recall'])} | "
              f"{pct(d['block_precision'])} | {pct(d['false_block_rate'])} |", "",
              "## Устойчивость к искажениям (recall нарушений / ложные блокировки)", "",
              "| искажение | с нормализацией | без нормализации |", "|---|---|---|"]
    for kind, row in r["robustness"]["results"].items():
        w, wo = row["with_normalization"], row["without_normalization"]
        lines.append(f"| {kind} | {pct(w['recall_violations'])} / {pct(w['false_block_rate'])} | "
                     f"{pct(wo['recall_violations'])} / {pct(wo['false_block_rate'])} |")
    lines += ["", "## Категории среди REVIEW + BLOCK", "",
              ", ".join(f"{k}: {pct(v)}" for k, v in r["categories_in_flagged"].items()), "",
              f"Офлайн-throughput полного пайплайна: **{r['throughput_offline_msgs_per_min']:,} сообщений/мин**", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
