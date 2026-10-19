"""Офлайн-бенчмарк скорости (без HTTP): throughput в сообщениях/мин и вклад каждого компонента."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.config import load_config, resolve  # noqa: E402
from moderation.data import load_split  # noqa: E402
from moderation.moderator import Moderator  # noqa: E402
from moderation.normalize import normalize  # noqa: E402


def timed(fn, *a):
    start = time.perf_counter()
    fn(*a)
    return time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5000)
    args = ap.parse_args()
    cfg = load_config()
    mod = Moderator(cfg)
    texts = load_split(cfg, "test")["text"].sample(frac=1, random_state=0).tolist()[: args.samples]
    mod.moderate_batch(texts[:64])  # прогрев

    result = {"samples": len(texts), "components": mod.components, "batch": {}}
    for bs in (1, 16, 64, 256):
        subset = texts[: min(len(texts), 1000 if bs == 1 else len(texts))]
        sec = timed(lambda: [mod.moderate_batch(subset[i:i + bs]) for i in range(0, len(subset), bs)])
        result["batch"][bs] = {"msgs_per_min": round(len(subset) / sec * 60),
                               "ms_per_msg": round(sec / len(subset) * 1000, 3)}
        print(f"batch={bs:4}: {result['batch'][bs]['msgs_per_min']:>10,} сообщений/мин")

    norm = [normalize(t) for t in texts]
    breakdown = {"normalize": timed(lambda: [normalize(t) for t in texts]),
                 "rules": timed(lambda: [mod.rules.match(t) for t in norm])}
    for name, model in mod.models.items():
        breakdown[name] = timed(model.predict_proba, norm)
    result["ms_per_msg_by_component"] = {k: round(v / len(texts) * 1000, 4) for k, v in breakdown.items()}
    print("мс на сообщение по компонентам:", result["ms_per_msg_by_component"])

    out = resolve(load_config().get("reports_dir", "reports"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
