"""Сводный отчёт reports/SUMMARY.md: итоговые метрики качества и производительности."""
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.config import load_config, resolve  # noqa: E402


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    cfg = load_config()
    rep = resolve(cfg.get("reports_dir", "reports"))
    m, bench = read(rep / "metrics.json"), read(rep / "benchmark.json")
    single, batch = read(rep / "loadtest_single.json"), read(rep / "loadtest_batch.json")
    pct = lambda v: f"{v:.1%}"
    lines = ["# Итоговые метрики", "", f"Окружение: {platform.platform()}, Python {platform.python_version()}", ""]
    if m:
        d, e = m["decisions"], m["components"]["ensemble"]
        lines += ["## Качество (тест)", "",
                  f"- ROC-AUC ансамбля: **{e['roc_auc']}**, PR-AUC: **{e['pr_auc']}**",
                  f"- Нарушения пойманы (BLOCK + REVIEW): **{pct(d['recall_violations'])}**, "
                  f"пропущено: **{pct(d['miss_rate'])}**",
                  f"- Доля REVIEW: **{pct(d['share_review'])}**, точность BLOCK: **{pct(d['block_precision'])}**, "
                  f"ложные блокировки: **{pct(d['false_block_rate'])}**", "",
                  "Подробно: [metrics.md](metrics.md), компромиссы порогов: [tradeoff.md](tradeoff.md)", ""]
    lines += ["## Производительность", ""]
    if single:
        lines.append(f"- API, одиночные запросы (`POST /moderate`, {single['concurrency']} параллельных клиентов): "
                     f"**{single['throughput_msgs_per_min']:,} messages/min**, p50 {single['latency_ms']['p50']} мс, "
                     f"p95 {single['latency_ms']['p95']} мс, ошибок {single['errors']}")
    if batch:
        lines.append(f"- API, батчи по {batch['batch_size']} (`POST /moderate/batch`): "
                     f"**{batch['throughput_msgs_per_min']:,} messages/min**")
    if bench:
        best = max(bench["batch"].values(), key=lambda b: b["msgs_per_min"])
        lines.append(f"- Офлайн, без HTTP: до **{best['msgs_per_min']:,} messages/min**; "
                     f"мс на сообщение по компонентам: {bench['ms_per_msg_by_component']}")
    (rep / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
