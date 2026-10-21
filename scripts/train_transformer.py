"""Дообучение rubert-tiny2 на бинарную классификацию → models/transformer/.

Устройство выбирается автоматически: CUDA → Apple MPS → CPU.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.config import load_config, resolve  # noqa: E402
from moderation.data import load_split, model_inputs, stratified_sample  # noqa: E402
from moderation.models.transformer import TransformerModel, pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", help="переопределить transformer.base_model")
    ap.add_argument("--max-train", type=int, help="переопределить transformer.max_train_samples")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()

    cfg = load_config()
    tcfg = cfg["transformer"]
    base = args.base_model or tcfg["base_model"]
    if resolve(base).exists():  # локальная папка с моделью
        base = str(resolve(base))
    epochs = args.epochs or tcfg["epochs"]
    seed = cfg["data"]["seed"]
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

    train = stratified_sample(load_split(cfg, "train"), args.max_train or tcfg["max_train_samples"], seed)
    val = stratified_sample(load_split(cfg, "val"), 20000, seed)
    x_train, y_train = model_inputs(train["text"], cfg), train["label"].tolist()
    x_val = model_inputs(val["text"], cfg)

    device = pick_device(tcfg.get("train_device", "auto"))
    print(f"база: {base} | устройство: {device} | train: {len(x_train)} | эпох: {epochs}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2).to(device)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=tcfg["max_length"],
                        return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc

    loader = DataLoader(list(zip(x_train, y_train)), batch_size=tcfg["batch_size"], shuffle=True,
                        collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=0.01)
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * total), total)

    out = resolve(tcfg["path"])
    start, step = time.time(), 0
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(), sched.step(), optim.zero_grad()
            step += 1
            if step % 100 == 0 or step == total:
                speed = step * tcfg["batch_size"] / (time.time() - start)
                print(f"эпоха {epoch + 1} шаг {step}/{total} loss {loss.item():.4f} | {speed:.0f} текстов/с")

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    model.eval()
    p = TransformerModel(out, tcfg.get("train_device", "auto"), tcfg["max_length"], tcfg["batch_size"]).predict_proba(
        x_val)
    info = {"base_model": base, "device": str(device), "train_samples": len(x_train), "epochs": epochs,
            "train_minutes": round((time.time() - start) / 60, 1),
            "val_roc_auc": round(float(roc_auc_score(val["label"], p)), 4),
            "val_pr_auc": round(float(average_precision_score(val["label"], p)), 4)}
    (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
