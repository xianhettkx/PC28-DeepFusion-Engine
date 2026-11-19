"""600-period strict temporal out-of-sample backtest with full metrics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, confusion_matrix,
    precision_recall_fscore_support
)
from features.engine import load_and_prepare, build_sequences_and_features, CLASSES
from models.deepfusion import DeepFusionNet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rolling_10000.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "evaluation" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def latest_model_path():
    pts = sorted(MODEL_DIR.glob("deepfusion_v*.pt"))
    return pts[-1] if pts else None

def run_backtest():
    df = load_and_prepare(DATA)
    seqs, feats, targets, issues = build_sequences_and_features(df)
    n = len(targets)
    test_size = 600
    train_end = n - test_size
    model_path = latest_model_path()
    if not model_path:
        raise FileNotFoundError("No trained model found. Run training first.")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = DeepFusionNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    seq_t = torch.from_numpy(seqs[train_end:])
    feat_t = torch.from_numpy(feats[train_end:])
    with torch.no_grad():
        logits = model(seq_t, feat_t)
        probs = torch.softmax(logits, dim=1).numpy()
    y_true = targets[train_end:]
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        ll = log_loss(y_true, probs, labels=[0,1,2,3])
    except Exception:
        ll = None
    brier = np.mean(np.sum((probs - np.eye(4)[y_true]) ** 2, axis=1))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3]).tolist()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], zero_division=0)
    train_labels = targets[:train_end]
    freq = np.bincount(train_labels, minlength=4) / len(train_labels)
    majority = int(np.argmax(freq))
    baseline_acc = (y_true == majority).mean()
    report = {
        "test_periods": int(test_size),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "log_loss": round(float(ll), 4) if ll is not None else None,
        "brier_score": round(float(brier), 4),
        "confusion_matrix": cm,
        "per_class": {CLASSES[i]: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4)} for i in range(4)},
        "baselines": {"majority_class_accuracy": round(float(baseline_acc), 4), "random_accuracy": 0.25},
        "note": "Strict temporal split. No future leakage. Lottery is random; results near baseline are expected."
    }
    out = REPORT_DIR / "backtest_600.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

if __name__ == "__main__":
    run_backtest()
