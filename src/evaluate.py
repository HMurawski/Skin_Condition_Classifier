import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from .config import BEST_WEIGHTS, CLASS_INDEX_PATH, DEVICE, PRED_THRESHOLD
from .data import get_dataloaders
from .model import build_model
from .utils import set_seed, get_device, load_checkpoint

@torch.no_grad()
def collect_logits(model, dataloader, device):
    """Collect true labels, predicted labels and max softmax probs."""
    model.eval()
    y_true, y_pred, maxp = [], [], []
    for xb, yb in dataloader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        p, pred = probs.max(dim=1)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        maxp.extend(p.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(maxp)

def evaluate_with_threshold(y_true, y_pred, maxp, classes, thr):
    """Metrics on confident subset (maxp >= thr) plus coverage."""
    mask = maxp >= thr
    cov = float(mask.mean())
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {"coverage": cov, "acc": 0.0, "f1_macro": 0.0, "report": "", "cm": None}
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro")
    rep = classification_report(yt, yp, target_names=classes, digits=4, zero_division=0)
    cm  = confusion_matrix(yt, yp, labels=list(range(len(classes))))
    return {"coverage": cov, "acc": acc, "f1_macro": f1m, "report": rep, "cm": cm, "n_conf": int(mask.sum()), "n_total": len(mask)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val","test"], default="val")
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds from 0.50 to 0.90")
    args = parser.parse_args()

    set_seed(42)
    device = get_device(DEVICE)

    train_loader, val_loader, test_loader, classes = get_dataloaders()
    loader = val_loader if args.split == "val" else test_loader

    model = build_model(num_classes=len(classes))
    model = load_checkpoint(model, str(BEST_WEIGHTS), device).to(device)

    y_true, y_pred, maxp = collect_logits(model, loader, device)

    if args.sweep:
        print("[SWEEP] threshold, coverage, acc_conf, f1_macro_conf")
        best = None
        for thr in np.arange(0.50, 0.91, 0.05):
            res = evaluate_with_threshold(y_true, y_pred, maxp, classes, thr)
            print(f"{thr:.2f}, {res['coverage']:.3f}, {res['acc']:.4f}, {res['f1_macro']:.4f}")
            # pick best by F1 on confident set (you can blend coverage if you want)
            if best is None or res["f1_macro"] > best["f1_macro"]:
                best = {"thr": thr, **res}
        print("\n[SWEEP] Suggested threshold by F1_macro:", best["thr"])
        return

    # Single-threshold evaluation (uses PRED_THRESHOLD from config)
    res = evaluate_with_threshold(y_true, y_pred, maxp, classes, PRED_THRESHOLD)
    print(f"[EVAL:{args.split}] threshold={PRED_THRESHOLD:.2f}")
    print(f"Coverage: {res['coverage']:.3f} ({res['n_conf']}/{res['n_total']})")
    print(f"Acc (confident): {res['acc']:.4f}  F1_macro (confident): {res['f1_macro']:.4f}")
    print("Classification report (confident only):")
    print(res["report"])
    print("Confusion matrix (confident only):")
    print(res["cm"])

if __name__ == "__main__":
    main()
