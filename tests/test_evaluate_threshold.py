import numpy as np
from src.evaluate import evaluate_with_threshold



def test_evaluate_threshold_basic():
    
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])
    maxp   = np.array([0.80, 0.60, 0.40])  # last uncertain at thr=0.5
    classes = ["a", "b", "c"]

    res = evaluate_with_threshold(y_true, y_pred, maxp, classes, thr=0.50)
    assert np.isclose(res["coverage"], 2 / 3)
    assert res["n_conf"] == 2 and res["n_total"] == 3
    #preds = [0,1], true = [0,1] -> acc=1.0
    assert np.isclose(res["acc"], 1.0)
    assert np.isclose(res["f1_macro"], 1.0)

def test_evaluate_threshold_no_confident():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    maxp   = np.array([0.1, 0.2])
    classes = ["a", "b"]
    
    res = evaluate_with_threshold(y_true, y_pred, maxp, classes, thr=0.9)
    
    assert np.isclose(res["coverage"], 0.0)
    assert res["n_conf"] == 0 and res["n_total"] == 2
    assert res["acc"] == 0.0 and res["f1_macro"] == 0.0 and res["cm"] is None