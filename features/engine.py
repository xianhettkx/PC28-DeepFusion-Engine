"""Feature engineering module. Strictly causal - only past information."""
from __future__ import annotations
import numpy as np
import pandas as pd

CLASSES = ["大单", "小单", "大双", "小双"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEQ_LEN = 32
FEATURE_DIM = 48

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["draw_nbr"] = pd.to_numeric(df["draw_nbr"]).astype(int)
    df = df.sort_values("draw_nbr").reset_index(drop=True)
    parsed = df["draw_number"].apply(lambda x: [int(p) for p in str(x).split("+")])
    df["n1"] = parsed.apply(lambda x: x[0])
    df["n2"] = parsed.apply(lambda x: x[1])
    df["n3"] = parsed.apply(lambda x: x[2])
    assert (df["n1"] + df["n2"] + df["n3"] == df["draw_num"]).all(), "Sum mismatch"
    df["label"] = df["combination_type"].map(CLASS_TO_IDX)
    return df

def build_sequences_and_features(df: pd.DataFrame):
    n = len(df)
    sums = df["draw_num"].values.astype(np.float32) / 27.0
    labels = df["label"].values.astype(np.int64)
    seqs, feats, targets, issues = [], [], [], []

    for i in range(SEQ_LEN, n):
        seqs.append(sums[i - SEQ_LEN:i])
        start = max(0, i - 50)
        past_sums = df["draw_num"].iloc[start:i].values
        past_labels = labels[start:i]
        feat = []
        for lag in [1, 2, 3, 5, 10]:
            feat.append(float(sums[i - lag]) if i - lag >= 0 else 0.0)
        feat.append(float(np.mean(past_sums) / 27.0) if len(past_sums) else 0.5)
        feat.append(float(np.std(past_sums) / 10.0) if len(past_sums) > 1 else 0.0)
        for c in range(4):
            feat.append(float(np.mean(past_labels == c)) if len(past_labels) else 0.25)
        feat.append(float(np.mean(past_sums >= 14)) if len(past_sums) else 0.5)
        feat.append(float(np.mean(past_sums % 2 == 1)) if len(past_sums) else 0.5)
        feat = (feat + [0.0] * FEATURE_DIM)[:FEATURE_DIM]
        feats.append(feat)
        targets.append(int(labels[i]))
        issues.append(int(df["draw_nbr"].iloc[i]))

    return (
        np.array(seqs, dtype=np.float32),
        np.array(feats, dtype=np.float32),
        np.array(targets, dtype=np.int64),
        np.array(issues, dtype=np.int64),
    )
