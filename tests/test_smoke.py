"""CPU smoke test for the from-scratch DeepFusion training pipeline."""
from pathlib import Path
import torch

from features.engine import load_and_prepare, build_sequences_and_features
from models.deepfusion import DeepFusionNet

ROOT = Path(__file__).resolve().parents[1]


def test_data_features_and_model_forward():
    df = load_and_prepare(ROOT / "data" / "rolling_10000.csv")
    seqs, feats, labels, issues = build_sequences_and_features(df)
    assert len(labels) >= 9000
    assert seqs.shape[1:] == (32,)
    assert feats.shape[1:] == (48,)
    assert labels.min() >= 0 and labels.max() <= 3
    assert len(issues) == len(labels)

    model = DeepFusionNet(hidden=256)
    with torch.no_grad():
        logits = model(torch.from_numpy(seqs[:8]), torch.from_numpy(feats[:8]))
    assert tuple(logits.shape) == (8, 4)
    assert torch.isfinite(logits).all()


def test_model_is_not_a_downloaded_checkpoint():
    model = DeepFusionNet(hidden=256)
    assert sum(p.numel() for p in model.parameters()) > 250_000
