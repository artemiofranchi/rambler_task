"""Загрузка и подготовка датасета → data/processed/{train,val,test}.parquet.

  python scripts/prepare_data.py                         # Hugging Face: AlexSham/Toxic_Russian_Comments
  python scripts/prepare_data.py --source file --path my.csv   # свой файл с колонками text,label
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.config import load_config, resolve  # noqa: E402
from moderation.normalize import normalize  # noqa: E402
from moderation.rules import RuleEngine  # noqa: E402


def load_source(args, cfg):
    if args.source == "hf":
        from datasets import load_dataset
        ds = load_dataset(cfg["data"]["hf_dataset"])
        return ds["train"].to_pandas(), ds["test"].to_pandas()
    path = Path(args.path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    train, test = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=cfg["data"]["seed"])
    return train, test


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["label"] = df["label"].astype(int)
    return df.drop_duplicates("text").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "file"], default="hf")
    ap.add_argument("--path", help="CSV/parquet с колонками text,label (для --source file)")
    ap.add_argument("--invert-labels", action="store_true", help="если в источнике 0 = токсично")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["data"]["seed"]

    train_raw, test_raw = load_source(args, cfg)
    n_raw = {"train": len(train_raw), "test": len(test_raw)}
    train, test = clean(train_raw), clean(test_raw)
    if args.invert_labels:
        train["label"], test["label"] = 1 - train["label"], 1 - test["label"]

    # утечка: одинаковые (после нормализации) тексты в train и test
    test_norm = set(test["text"].map(normalize))
    leak = train["text"].map(normalize).isin(test_norm)
    train = train[~leak].reset_index(drop=True)

    # проверка направления меток: у токсичного класса однозначных правил должно срабатывать больше
    rules = RuleEngine()
    sample = train.sample(min(len(train), 20000), random_state=seed)
    strong = sample["text"].map(lambda t: any(h.strong for h in rules.match(normalize(t))))
    rate = strong.groupby(sample["label"]).mean().to_dict()
    if rate.get(0, 0) > rate.get(1, 0):
        sys.exit(f"Похоже, метки инвертированы (доля мата: label0={rate.get(0):.3f}, label1={rate.get(1):.3f}). "
                 "Перезапустите с --invert-labels.")

    train, val = train_test_split(train, test_size=cfg["data"]["val_size"], stratify=train["label"],
                                  random_state=seed)
    out = resolve(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stats = {"source": cfg["data"]["hf_dataset"] if args.source == "hf" else str(args.path), "raw": n_raw,
             "removed_train_test_leak": int(leak.sum()), "strong_rule_rate_by_label": rate}
    for name, df in (("train", train), ("val", val), ("test", test)):
        df.reset_index(drop=True).to_parquet(out / f"{name}.parquet", index=False)
        stats[name] = {"rows": len(df), "toxic_share": round(float(df["label"].mean()), 4)}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
