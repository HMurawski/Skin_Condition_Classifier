import csv
import os
from collections import Counter

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import BEST_WEIGHTS, CLASS_INDEX_PATH, DEVICE, EPOCHS, LR, SEED, WEIGHT_DECAY
from .data import get_dataloaders
from .logging_utils import init_logger
from .model import build_model
from .utils import get_device, save_checkpoint, set_seed


@torch.no_grad()
def validate(model: torch.nn.Module, loader, device: torch.device):
    """Validation loop returning accuracy and macro-F1."""
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(1)
        y_true.extend(yb.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m


def main():
    # Logging
    log = init_logger(__name__)

    # Seed + device
    set_seed(SEED)
    device = get_device(DEVICE)

    # Data
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    # Compute class weights from the training set
    counts = Counter([label for _, label in train_loader.dataset.samples])
    num_classes = len(train_loader.dataset.classes)
    freq = torch.tensor([counts[i] for i in range(num_classes)], dtype=torch.float)

    # Inverse-frequency weights normalized to sum to num_classes
    weights = 1.0 / (freq + 1e-6)
    weights = num_classes * (weights / weights.sum())
    weights = weights.to(device)
    log.info("Class counts: %s", freq.tolist())
    log.info("Class weights: %s", weights.tolist())

    # Save classes order for UI
    CLASS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_INDEX_PATH, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    log.info("Classes: %s", classes)

    # Model + loss + optimizer + scheduler
    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    metrics_path = "artifacts/metrics.csv"
    os.makedirs("artifacts", exist_ok=True)
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_acc", "val_f1", "lr"])

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach()))

        val_acc, val_f1 = validate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        log.info("VAL acc=%.4f  f1_macro=%.4f  lr=%.6f", val_acc, val_f1, lr)

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{val_acc:.6f}", f"{val_f1:.6f}", f"{lr:.8f}"])

        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, str(BEST_WEIGHTS))
            log.info("Saved best to %s (f1_macro=%.4f)", BEST_WEIGHTS, best_f1)


if __name__ == "__main__":
    main()
