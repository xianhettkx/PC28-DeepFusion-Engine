"""Generate a recursively updated, immutable 20-period forecast plan."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from features.engine import load_and_prepare,CLASSES
from models.deepfusion import DeepFusionNet

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"rolling_10000.csv"
PLAN_DIR=ROOT/"prediction"/"plans"
MODEL_DIR=ROOT/"models"

def make_features(sums,labels):
    past=np.asarray(sums[-50:],dtype=np.float32)
    labs=np.asarray(labels[-50:],dtype=np.int64)
    feat=[]
    for lag in [1,2,3,5,10]:
        feat.append(float(sums[-lag])/27.0 if len(sums)>=lag else 0.0)
    feat.extend([float(past.mean()/27.0),float(past.std()/10.0) if len(past)>1 else 0.0])
    for c in range(4): feat.append(float(np.mean(labs==c)) if len(labs) else 0.25)
    feat.append(float(np.mean(past>=14)) if len(past) else 0.5)
    feat.append(float(np.mean(past%2==1)) if len(past) else 0.5)
    return np.asarray((feat+[0.0]*48)[:48],dtype=np.float32)

def generate_plan(plan_id:str|None=None):
    df=load_and_prepare(DATA)
    if len(df)<100: raise RuntimeError("Not enough history to generate a forecast")
    latest_issue=int(df["draw_nbr"].iloc[-1])
    model_path=sorted(MODEL_DIR.glob("deepfusion_v*.pt"))[-1]
    ckpt=torch.load(model_path,map_location="cpu",weights_only=False)
    hidden=int(ckpt.get("model_config",{}).get("hidden",512))
    model=DeepFusionNet(hidden=hidden)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    sums=[float(x) for x in df["draw_num"].tolist()]
    labels=[int(x) for x in df["label"].tolist()]
    seq=np.asarray(sums[-32:],dtype=np.float32)/27.0
    # Historical class-conditioned mean sum is used only as a transparent
    # recursive expectation, never as a claim about the unknown actual draw.
    class_means={}
    for i,name in enumerate(CLASSES):
        vals=[s for s,l in zip(sums,labels) if l==i]
        class_means[i]=float(np.mean(vals)) if vals else float(np.mean(sums))
    predictions=[]
    for step in range(1,21):
        feat=make_features(sums,labels)
        with torch.no_grad():
            logits=model(torch.from_numpy(seq[None]),torch.from_numpy(feat[None]))
            probs=torch.softmax(logits,dim=1).numpy()[0]
        idx=int(np.argmax(probs))
        expected_sum=class_means[idx]
        predictions.append({
          "issue":latest_issue+step,
          "prediction":CLASSES[idx],
          "probabilities":{CLASSES[i]:round(float(probs[i]),6) for i in range(4)},
          "recursive_expected_sum":round(expected_sum,4)
        })
        sums.append(expected_sum); labels.append(idx)
        seq=np.concatenate([seq[1:],np.asarray([expected_sum/27.0],dtype=np.float32)])
    plan_id=plan_id or f"DF-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    record={
      "plan_id":plan_id,"generated_at":datetime.now(timezone.utc).isoformat(),
      "model_version":model_path.stem,"last_historical_issue":latest_issue,
      "forecast_method":"recursive_expected_value_feedback",
      "predictions":predictions,"frozen":True
    }
    canonical=json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    record["sha256"]=hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    PLAN_DIR.mkdir(parents=True,exist_ok=True)
    out=PLAN_DIR/f"{plan_id}.json"
    if out.exists(): raise FileExistsError(f"Refusing to overwrite frozen plan: {out}")
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"saved":str(out.relative_to(ROOT)),"sha256":record["sha256"],"forecast_periods":len(predictions),"first_issue":predictions[0]["issue"],"last_issue":predictions[-1]["issue"]},ensure_ascii=False))
    return record

if __name__=="__main__": generate_plan()
