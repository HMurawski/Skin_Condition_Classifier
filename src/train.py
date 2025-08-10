# src/train.py
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

from .config import (
    SEED, DEVICE, LR, WEIGHT_DECAY, EPOCHS,
    BEST_WEIGHTS, CLASS_INDEX_PATH
)
from .data import get_dataloaders
from .model import build_model
from .utils import set_seed, get_device, save_checkpoint

@torch.no_grad()
def validate(model, loader, device):
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
    f1m = f1_score(y_true, y_pred, average='macro')
    return acc, f1m

def main():
    # Seed + device
    set_seed(SEED)
    device = get_device(DEVICE)

    # Data
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    # Compute class weights from the training set (ImageFolder stores (path, class_idx))
    counts = Counter([label for _, label in train_loader.dataset.samples])
    num_classes = len(train_loader.dataset.classes)
    freq = torch.tensor([counts[i] for i in range(num_classes)], dtype=torch.float)
    
    # Inverse-frequency weights normalized to sum to num_classes (stability)
    weights = (1.0 / (freq + 1e-6))
    weights = num_classes * (weights / weights.sum())
    weights = weights.to(device)
    print(f"[INFO] Class counts: {freq.tolist()}")
    print(f"[INFO] Class weights: {weights.tolist()}")
    
    
    # Save classes order (for inference/Streamlit)
    CLASS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_INDEX_PATH, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    print(f"[INFO] Classes: {classes}")

    # Model + loss + optimizer + scheduler
    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)  # no verbose for compatibility

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach()))

        val_acc, val_f1 = validate(model, val_loader, device)
        print(f"[VAL] acc={val_acc:.4f}  f1_macro={val_f1:.4f}")
        scheduler.step(val_f1)
        print(f"[INFO] Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, str(BEST_WEIGHTS))
            print(f"Saved best to {BEST_WEIGHTS} (f1_macro={best_f1:.4f})")

   

if __name__ == "__main__":
    main()
