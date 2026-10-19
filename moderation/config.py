"""Загрузка конфигурации: YAML + переопределения из переменных окружения."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"

# Переменная окружения → (секция, ключ, тип)
ENV_OVERRIDES = {
    "MODERATION_DEVICE": ("transformer", "device", str),
    "MODERATION_USE_TRANSFORMER": ("transformer", "enabled", lambda v: v.lower() in ("1", "true", "yes")),
    "MODERATION_TORCH_THREADS": ("transformer", "num_threads", int),
    "MODERATION_TARGET_RECALL": ("policy", "target_recall", float),
    "MODERATION_TARGET_BLOCK_PRECISION": ("policy", "target_block_precision", float),
    "MODERATION_BATCH_MAX_SIZE": ("service", "batch_max_size", int),
    "MODERATION_BATCH_MAX_WAIT_MS": ("service", "batch_max_wait_ms", float),
}


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.getenv("MODERATION_CONFIG") or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for env, (section, key, cast) in ENV_OVERRIDES.items():
        if os.getenv(env) not in (None, ""):
            cfg[section][key] = cast(os.environ[env])
    return cfg


def resolve(p: str | os.PathLike) -> Path:
    """Пути из конфига считаются от корня репозитория."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
