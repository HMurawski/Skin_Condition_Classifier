import numpy as np
from src.evaluate import evaluate_with_threshold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def evaluate_with_threshold(y_true, y_pred, maxp, classes, thr):
    mask = maxp >= thr
    cov = float(mask.mean())
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {"coverage": cov, "acc": 0.0, "f1_macro": 0.0, "report": "", "cm": None}

    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro")

    all_labels = list(range(len(classes))) 
    rep = classification_report(
        yt, yp,
        labels=all_labels,               
        target_names=classes,
        digits=4,
        zero_division=0
    )
    cm  = confusion_matrix(yt, yp, labels=all_labels)
    return {"coverage": cov, "acc": acc, "f1_macro": f1m, "report": rep, "cm": cm,
            "n_conf": int(mask.sum()), "n_total": len(mask)}

def test_evaluate_threshold_basic():
    
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])
    maxp   = np.array([0.80, 0.60, 0.40])  # last uncertain at thr=0.5
    classes = ["a", "b", "c"]

    res = evaluate_with_threshold(y_true, y_pred, maxp, classes, thr=0.50)
    assert res["coverage"] == 2/3
    assert res["n_conf"] == 2 and res["n_total"] == 3
    #preds = [0,1], true = [0,1] -> acc=1.0
    assert res["acc"] == 1.0

def test_evaluate_threshold_no_confident():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    maxp   = np.array([0.1, 0.2])
    classes = ["a", "b"]
    res = evaluate_with_threshold(y_true, y_pred, maxp, classes, thr=0.9)
    assert res["coverage"] == 0.0
    assert res["acc"] == 0.0 and res["f1_macro"] == 0.0 and res["cm"] is None
