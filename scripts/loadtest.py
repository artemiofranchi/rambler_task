"""Нагрузочный тест работающего API → throughput (сообщений/мин), задержки p50/p95/p99, доля ошибок.

  python scripts/loadtest.py --url http://localhost:8000 --concurrency 32 --duration 30
  python scripts/loadtest.py --mode batch --batch-size 64      # через /moderate/batch
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.config import load_config, resolve  # noqa: E402


def load_texts(n=5000):
    try:
        import pandas as pd
        path = resolve(load_config()["data"]["processed_dir"]) / "test.parquet"
        return pd.read_parquet(path)["text"].sample(frac=1, random_state=0).tolist()[:n]
    except Exception:
        sample = resolve("data/samples/messages.jsonl")
        return [json.loads(line)["text"] for line in sample.read_text(encoding="utf-8").splitlines() if line]


async def worker(client, args, texts, stop_at, lat, counters):
    rng = random.Random()
    while time.perf_counter() < stop_at:
        if args.mode == "single":
            url, payload, n = "/moderate", {"text": rng.choice(texts)}, 1
        else:
            batch = rng.sample(texts, min(args.batch_size, len(texts)))
            url, payload, n = "/moderate/batch", {"texts": batch}, len(batch)
        start = time.perf_counter()
        try:
            r = await client.post(url, json=payload)
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        lat.append(time.perf_counter() - start)
        counters["messages" if ok else "errors"] += n


async def main_async(args):
    texts = load_texts()
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=30, limits=limits) as client:
        (await client.get("/health")).raise_for_status()
        await asyncio.gather(*[worker(client, args, texts, time.perf_counter() + 2, [], {"messages": 0, "errors": 0})
                               for _ in range(args.concurrency)])  # прогрев 2 с
        lat, counters = [], {"messages": 0, "errors": 0}
        start = time.perf_counter()
        await asyncio.gather(*[worker(client, args, texts, start + args.duration, lat, counters)
                               for _ in range(args.concurrency)])
        elapsed = time.perf_counter() - start
    ms = np.array(lat) * 1000
    result = {"mode": args.mode, "concurrency": args.concurrency, "duration_s": round(elapsed, 1),
              "batch_size": args.batch_size if args.mode == "batch" else 1,
              "messages": counters["messages"], "errors": counters["errors"],
              "throughput_msgs_per_min": round(counters["messages"] / elapsed * 60),
              "requests_per_s": round(len(lat) / elapsed, 1),
              "latency_ms": {"p50": round(float(np.percentile(ms, 50)), 1),
                             "p95": round(float(np.percentile(ms, 95)), 1),
                             "p99": round(float(np.percentile(ms, 99)), 1)}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = resolve(load_config().get("reports_dir", "reports"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"loadtest_{args.mode}.json").write_text(json.dumps(result, indent=2))
    print(f"\nTHROUGHPUT: {result['throughput_msgs_per_min']:,} messages/min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--mode", choices=["single", "batch"], default="single")
    ap.add_argument("--batch-size", type=int, default=64)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
