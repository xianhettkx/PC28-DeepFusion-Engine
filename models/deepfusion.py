"""DeepFusion model definition."""
import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden // 2, num_layers=2, batch_first=True, dropout=0.1)
        self.proj = nn.Linear(hidden // 2, hidden // 2)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.proj(out[:, -1])

class FeatureEncoder(nn.Module):
    def __init__(self, in_dim=48, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class DeepFusionNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.seq_enc = SequenceEncoder(hidden)
        self.feat_enc = FeatureEncoder(hidden=hidden)
        self.fusion = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 4),
        )
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, seq, feat):
        s = self.seq_enc(seq)
        f = self.feat_enc(feat)
        logits = self.fusion(torch.cat([s, f], dim=-1))
        return logits / self.temp.clamp(min=0.1)
