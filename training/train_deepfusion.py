"""Train the project-owned DeepFusion network from random initialization."""
from __future__ import annotations
import argparse, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from features.engine import load_and_prepare, build_sequences_and_features
from models.deepfusion import DeepFusionNet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rolling_10000.csv"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
HIDDEN = 512
SEED = 20261120

def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def fit_model(model, seqs, feats, labels, epochs=1, batch_size=128, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ds = TensorDataset(torch.from_numpy(seqs).float(), torch.from_numpy(feats).float(), torch.from_numpy(labels).long())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history=[]
    for epoch in range(epochs):
        model.train(); total=0.0; count=0
        for seq, feat, target in loader:
            seq, feat, target = seq.to(device), feat.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(seq,feat); loss=criterion(logits,target)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += float(loss.item())*len(target); count += len(target)
        history.append(total/max(count,1))
    return model, history

def evaluate(model, seqs, feats, labels):
    model.eval()
    with torch.no_grad():
        logits=model(torch.from_numpy(seqs).float(),torch.from_numpy(feats).float())
        loss=float(nn.functional.cross_entropy(logits,torch.from_numpy(labels).long()).item())
        pred=logits.argmax(1).numpy()
    return loss, float((pred==labels).mean())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--if-new-20",action="store_true")
    parser.add_argument("--epochs",type=int,default=18)
    args=parser.parse_args()
    set_seed()
    df=load_and_prepare(DATA)
    seqs,feats,labels,issues=build_sequences_and_features(df)
    if len(labels)<1000: raise RuntimeError(f"Not enough training samples: {len(labels)}")
    latest_issue=int(issues[-1])
    MODEL_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)
    old=sorted(MODEL_DIR.glob("deepfusion_v*.pt"))
    if args.if_new_20 and old:
        previous=torch.load(old[-1],map_location="cpu",weights_only=False)
        trained=int(previous.get("trained_through_issue",0))
        new_count=int(np.sum(issues>trained))
        if new_count<20:
            print(json.dumps({"status":"skip","reason":"fewer than 20 new draws","new_draws":new_count,"trained_through_issue":trained,"latest_issue":latest_issue}))
            return
    split=max(int(len(labels)*0.85),1)
    if len(labels)-split<100: split=len(labels)-100
    xtr, ftr, ytr=seqs[:split],feats[:split],labels[:split]
    xva, fva, yva=seqs[split:],feats[split:],labels[split:]
    model=DeepFusionNet(hidden=HIDDEN)
    initial_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    model, losses=fit_model(model,xtr,ftr,ytr,epochs=max(1,args.epochs))
    val_loss,val_acc=evaluate(model,xva,fva,yva)
    version=f"deepfusion_v{latest_issue}"
    path=MODEL_DIR/f"{version}.pt"
    checkpoint={
      "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
      "model_config":{"hidden":HIDDEN,"sequence_length":32,"feature_dim":48,"classes":4},
      "trained_through_issue":latest_issue,
      "trained_at":datetime.now(timezone.utc).isoformat(),
      "parameter_count":initial_params,
      "train_samples":int(len(ytr)),"validation_samples":int(len(yva)),
      "epochs":len(losses),"train_loss":losses[-1],"validation_loss":val_loss,
      "validation_accuracy":val_acc,"seed":SEED,"from_random_initialization":True
    }
    torch.save(checkpoint,path)
    size=path.stat().st_size
    if size<=1_000_000: raise RuntimeError(f"Model checkpoint too small: {size} bytes")
    report={**{k:v for k,v in checkpoint.items() if k!="model_state"},"model_file":str(path.relative_to(ROOT)),"model_size_bytes":size,"status":"trained"}
    (LOG_DIR/"training_latest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
