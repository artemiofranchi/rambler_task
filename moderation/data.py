"""Работа с подготовленными данными (data/processed/{train,val,test}.parquet: text, label)."""
import time

import pandas as pd

from moderation.config import resolve
from moderation.normalize import normalize


def load_split(cfg: dict, split: str) -> pd.DataFrame:
    path = resolve(cfg["data"]["processed_dir"]) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} не найден — сначала выполните `make data`")
    return pd.read_parquet(path)


def model_inputs(texts, cfg: dict) -> list[str]:
    """То же преобразование, что делает Moderator перед моделями."""
    if cfg["normalization"]["enabled"]:
        return [normalize(t) for t in texts]
    return [t.lower() for t in texts]


def stratified_sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if not n or n >= len(df):
        return df
    from sklearn.model_selection import train_test_split
    sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sample.reset_index(drop=True)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
