"""600-draw prequential walk-forward test; every block only sees past outcomes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,log_loss,confusion_matrix,precision_recall_fscore_support
from features.engine import load_and_prepare,build_sequences_and_features,CLASSES
from models.deepfusion import DeepFusionNet
from training.train_deepfusion import fit_model

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"rolling_10000.csv"
REPORT_DIR=ROOT/"evaluation"/"reports"

def run_backtest():
    df=load_and_prepare(DATA)
    seqs,feats,targets,issues=build_sequences_and_features(df)
    test_size=600
    if len(targets)<=test_size+1000: raise RuntimeError("Insufficient pre-test history")
    cutoff=len(targets)-test_size
    torch.manual_seed(20261120)
    model=DeepFusionNet(hidden=512)
    model,_=fit_model(model,seqs[:cutoff],feats[:cutoff],targets[:cutoff],epochs=12,lr=2e-4)
    all_probs=[]; all_true=[]; rows=[]
    block=20
    for start in range(cutoff,len(targets),block):
        end=min(start+block,len(targets))
        model.eval()
        with torch.no_grad():
            logits=model(torch.from_numpy(seqs[start:end]).float(),torch.from_numpy(feats[start:end]).float())
            probs=torch.softmax(logits,dim=1).numpy()
        all_probs.append(probs); all_true.extend(targets[start:end].tolist())
        for j,p in enumerate(probs,start):
            idx=int(np.argmax(p))
            rows.append({"issue":int(issues[j]),"prediction":CLASSES[idx],"probabilities":{CLASSES[k]:round(float(p[k]),6) for k in range(4)},"actual":CLASSES[int(targets[j])],"hit":bool(idx==int(targets[j]))})
        # Only after scoring this block do its actual outcomes enter training.
        model,_=fit_model(model,seqs[:end],feats[:end],targets[:end],epochs=1,lr=5e-5)
    probs=np.concatenate(all_probs,axis=0); y=np.asarray(all_true,dtype=np.int64); pred=probs.argmax(1)
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1,2,3],zero_division=0)
    cm=confusion_matrix(y,pred,labels=[0,1,2,3]).tolist()
    freq=np.bincount(targets[:cutoff],minlength=4); baseline=int(freq.argmax())
    report={
      "method":"prequential_walk_forward_20_draw_blocks",
      "test_periods":int(len(y)),"first_test_issue":int(issues[cutoff]),"last_test_issue":int(issues[-1]),
      "accuracy":round(float(accuracy_score(y,pred)),6),
      "macro_f1":round(float(f1_score(y,pred,average="macro",zero_division=0)),6),
      "log_loss":round(float(log_loss(y,probs,labels=[0,1,2,3])),6),
      "brier_score":round(float(np.mean(np.sum((probs-np.eye(4)[y])**2,axis=1))),6),
      "confusion_matrix":cm,
      "per_class":{CLASSES[i]:{"precision":round(float(precision[i]),6),"recall":round(float(recall[i]),6),"f1":round(float(f1[i]),6)} for i in range(4)},
      "baselines":{"initial_majority_class":CLASSES[baseline],"initial_majority_accuracy":round(float(np.mean(y==baseline)),6),"uniform_random_accuracy":0.25},
      "predictions":rows,
      "note":"Predictions are made in chronological 20-draw blocks. Outcomes from a block are added only after that block is scored. No future labels enter training."
    }
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    (REPORT_DIR/"backtest_600.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="predictions"},ensure_ascii=False,indent=2))
    return report

if __name__=="__main__": run_backtest()
