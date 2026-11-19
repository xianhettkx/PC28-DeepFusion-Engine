"""20-period frozen prediction plan generator."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from features.engine import load_and_prepare, build_sequences_and_features, CLASSES
from models.deepfusion import DeepFusionNet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rolling_10000.csv"
PLAN_DIR = ROOT / "prediction" / "plans"
PLAN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models"

def generate_plan(plan_id: str | None = None):
    df = load_and_prepare(DATA)
    seqs, feats, targets, issues = build_sequences_and_features(df)
    last_issue = int(issues[-1])
    model_path = sorted(MODEL_DIR.glob("deepfusion_v*.pt"))[-1]
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = DeepFusionNet()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    plans = []
    current_seq = seqs[-1].copy()
    current_feat = feats[-1].copy()
    for step in range(20):
        with torch.no_grad():
            logits = model(torch.from_numpy(current_seq[None]), torch.from_numpy(current_feat[None]))
            probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(probs.argmax())
        plans.append({
            "period_offset": step + 1,
            "predicted_class": CLASSES[pred_idx],
            "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(4)},
        })
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = 0.5
    plan_id = plan_id or f"DF-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    record = {
        "plan_id": plan_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_path.stem,
        "last_historical_issue": last_issue,
        "predictions": plans,
        "frozen": True,
        "note": "Plan is frozen. Predictions must not be modified after generation."
    }
    out = PLAN_DIR / f"{plan_id}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Plan saved: {out}")
    return record

if __name__ == "__main__":
    generate_plan()
