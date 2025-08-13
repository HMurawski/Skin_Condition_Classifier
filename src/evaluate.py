import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from .config import BEST_WEIGHTS, DEVICE, PRED_THRESHOLD
from .data import get_dataloaders
from .model import build_model
from .utils import set_seed, get_device, load_checkpoint
from .logging_utils import init_logger


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
    return {
        "coverage": cov,
        "acc": acc,
        "f1_macro": f1m,
        "report": rep,
        "cm": cm,
        "n_conf": int(mask.sum()),
        "n_total": len(mask),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds from 0.50 to 0.90")
    args = parser.parse_args()

    # Logger + seed/device
    log = init_logger(__name__)
    set_seed(42)
    device = get_device(DEVICE)
    log.info("Using device: %s", device)

    # Data & model
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    loader = val_loader if args.split == "val" else test_loader

    model = build_model(num_classes=len(classes))
    model = load_checkpoint(model, str(BEST_WEIGHTS), device).to(device)

    y_true, y_pred, maxp = collect_logits(model, loader, device)

    if args.sweep:
        log.info("[SWEEP] threshold, coverage, acc_conf, f1_macro_conf")
        best = None
        for thr in np.arange(0.50, 0.91, 0.05):
            res = evaluate_with_threshold(y_true, y_pred, maxp, classes, thr)
            log.info("%.2f, %.3f, %.4f, %.4f", thr, res["coverage"], res["acc"], res["f1_macro"])
            if best is None or res["f1_macro"] > best["f1_macro"]:
                best = {"thr": thr, **res}
        if best is not None:
            log.info("[SWEEP] Suggested threshold by F1_macro: %.2f", best["thr"])
        else:
            log.warning("[SWEEP] No confident samples across thresholds.")
        return

    # Single-threshold evaluation
    res = evaluate_with_threshold(y_true, y_pred, maxp, classes, PRED_THRESHOLD)
    log.info("[EVAL:%s] threshold=%.2f", args.split, PRED_THRESHOLD)
    log.info("Coverage: %.3f (%d/%d)", res["coverage"], res.get("n_conf", 0), res.get("n_total", 0))
    log.info("Acc (confident): %.4f  F1_macro (confident): %.4f", res["acc"], res["f1_macro"])
    log.info("Classification report (confident only):\n%s", res["report"])
    log.info("Confusion matrix (confident only):\n%s", res["cm"])


if __name__ == "__main__":
    main()
