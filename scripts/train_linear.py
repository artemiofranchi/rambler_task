"""Обучение TF-IDF + LogReg → models/linear.joblib."""
import sys
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.config import load_config, resolve  # noqa: E402
from moderation.data import Timer, load_split, model_inputs  # noqa: E402
from moderation.models.linear import LinearModel  # noqa: E402


def main():
    cfg = load_config()
    train, val = load_split(cfg, "train"), load_split(cfg, "val")
    with Timer() as t_prep:
        x_train, x_val = model_inputs(train["text"], cfg), model_inputs(val["text"], cfg)
    print(f"нормализация: {t_prep.seconds:.1f} c")

    with Timer() as t_fit:
        model = LinearModel.build(cfg["linear"]).fit(x_train, train["label"])
    p = model.predict_proba(x_val)
    print(f"обучение: {t_fit.seconds:.1f} c на {len(train)} текстах")
    print(f"val ROC-AUC = {roc_auc_score(val['label'], p):.4f}, PR-AUC = {average_precision_score(val['label'], p):.4f}")

    path = resolve(cfg["linear"]["path"])
    model.save(path)
    print(f"сохранено: {path} ({path.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
