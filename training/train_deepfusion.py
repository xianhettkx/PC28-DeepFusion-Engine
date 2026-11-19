"""Train DeepFusion; warm-start later versions and reject non-improving candidates."""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from features.engine import build_sequences_and_features, load_and_prepare
from models.deepfusion import DeepFusionNet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rolling_10000.csv"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
HIDDEN = 256
SEED = 20261120


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_issue(path: Path) -> int:
    match = re.fullmatch(r"deepfusion_v(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint() -> Path | None:
    files = [p for p in MODEL_DIR.glob("deepfusion_v*.pt") if checkpoint_issue(p) >= 0]
    return max(files, key=checkpoint_issue) if files else None


def fit_model(model, seqs, feats, labels, epochs=1, batch_size=128, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ds = TensorDataset(
        torch.from_numpy(seqs).float(),
        torch.from_numpy(feats).float(),
        torch.from_numpy(labels).long(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for seq, feat, target in loader:
            seq, feat, target = seq.to(device), feat.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(seq, feat)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(target)
            count += len(target)
        history.append(total / max(count, 1))
    return model, history


def evaluate(model, seqs, feats, labels):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        seq = torch.from_numpy(seqs).float().to(device)
        feat = torch.from_numpy(feats).float().to(device)
        target = torch.from_numpy(labels).long().to(device)
        logits = model(seq, feat)
        loss = float(nn.functional.cross_entropy(logits, target).item())
        pred = logits.argmax(1).detach().cpu().numpy()
    return loss, float((pred == labels).mean())


def write_report(report: dict):
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "training_latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-new-20", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints; reserved for clean CI validation.")
    parser.add_argument("--min-improvement", type=float, default=1e-5)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.min_improvement < 0:
        raise ValueError("--min-improvement cannot be negative")

    set_seed()
    df = load_and_prepare(DATA)
    seqs, feats, labels, issues = build_sequences_and_features(df)
    if len(labels) < 1000:
        raise RuntimeError(f"Not enough training samples: {len(labels)}")

    latest_issue = int(issues[-1])
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    previous_path = None if args.fresh else latest_checkpoint()
    previous = None

    if previous_path is not None:
        previous = torch.load(previous_path, map_location="cpu", weights_only=False)
        trained_through = int(previous.get("trained_through_issue", 0))
        new_count = int(np.sum(issues > trained_through))
        if args.if_new_20 and new_count < 20:
            print(json.dumps({
                "status": "skip",
                "reason": "fewer than 20 new draws",
                "new_draws": new_count,
                "trained_through_issue": trained_through,
                "latest_issue": latest_issue,
            }, ensure_ascii=False))
            return
    elif args.if_new_20:
        print(json.dumps({
            "status": "skip",
            "reason": "no previous checkpoint; run initial training first",
            "latest_issue": latest_issue,
        }, ensure_ascii=False))
        return

    split = max(int(len(labels) * 0.85), 1)
    if len(labels) - split < 100:
        split = len(labels) - 100
    xtr, ftr, ytr = seqs[:split], feats[:split], labels[:split]
    xva, fva, yva = seqs[split:], feats[split:], labels[split:]

    model = DeepFusionNet(hidden=HIDDEN)
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    baseline_loss = None
    baseline_accuracy = None
    if previous is not None:
        old_hidden = int(previous.get("model_config", {}).get("hidden", HIDDEN))
        if old_hidden != HIDDEN:
            raise RuntimeError(f"Checkpoint hidden size {old_hidden} != current {HIDDEN}")
        model.load_state_dict(previous["model_state"])
        baseline_loss, baseline_accuracy = evaluate(model, xva, fva, yva)
        candidate = copy.deepcopy(model)
    else:
        candidate = model

    candidate, losses = fit_model(
        candidate, xtr, ftr, ytr, epochs=args.epochs
    )
    val_loss, val_acc = evaluate(candidate, xva, fva, yva)
    improvement = None if baseline_loss is None else baseline_loss - val_loss

    common = {
        "model_config": {
            "hidden": HIDDEN, "sequence_length": 32,
            "feature_dim": 48, "classes": 4,
        },
        "trained_through_issue": latest_issue,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "parameter_count": initial_params,
        "train_samples": int(len(ytr)),
        "validation_samples": int(len(yva)),
        "epochs": len(losses),
        "train_loss": losses[-1],
        "validation_loss": val_loss,
        "validation_accuracy": val_acc,
        "baseline_validation_loss": baseline_loss,
        "baseline_validation_accuracy": baseline_accuracy,
        "validation_loss_improvement": improvement,
        "min_required_improvement": args.min_improvement,
        "initialized_from_checkpoint": str(previous_path.relative_to(ROOT)) if previous_path else None,
        "from_random_initialization": previous is None,
        "seed": SEED,
    }

    if previous is not None and (improvement is None or improvement < args.min_improvement):
        report = {
            **common,
            "model_file": str(previous_path.relative_to(ROOT)),
            "model_size_bytes": previous_path.stat().st_size,
            "status": "rejected",
            "reason": "candidate did not improve chronological validation loss enough",
        }
        write_report(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    version = f"deepfusion_v{latest_issue}"
    path = MODEL_DIR / f"{version}.pt"
    checkpoint = {
        **common,
        "model_state": {k: v.detach().cpu() for k, v in candidate.state_dict().items()},
    }
    torch.save(checkpoint, path)
    size = path.stat().st_size
    if size <= 1_000_000:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Model checkpoint too small: {size} bytes")

    report = {
        **common,
        "model_file": str(path.relative_to(ROOT)),
        "model_size_bytes": size,
        "status": "trained",
    }
    write_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
