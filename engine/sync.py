#!/usr/bin/env python3
"""Idempotent PC28 history bootstrap + API synchronizer. Standard-library only."""
from __future__ import annotations
import csv, json, os, re, sqlite3, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "pc28.sqlite3"
API_URL = os.getenv("PC28_API_URL", "https://pc28.help/api/kj.json?nbr=500")
ROLLING_SIZE = int(os.getenv("PC28_ROLLING_SIZE", "10000"))
FIELDS = ["draw_nbr","draw_date","draw_time","draw_number","draw_num","size_type","parity_type","combination_type"]
ALIASES = {
 "draw_nbr": ("draw_nbr","expect","issue","period","qihao","drawno","draw_no","number_id"),
 "draw_date": ("draw_date","date","open_date"),
 "draw_time": ("draw_time","time","open_time"),
 "draw_number": ("draw_number","opencode","open_code","code","result","open_result"),
 "draw_num": ("draw_num","sum","total","和值"),
}
def first(obj, names):
    for name in names:
        if name in obj and obj[name] not in (None, ""):
            return obj[name]
    return None

def extract_rows(payload):
    if isinstance(payload, list): return payload
    if not isinstance(payload, dict): raise ValueError("API JSON root must be an object or array")
    for key in ("data","list","rows","result","records","history"):
        value = payload.get(key)
        if isinstance(value, list): return value
        if isinstance(value, dict):
            for nested in ("data","list","rows","records"):
                if isinstance(value.get(nested), list): return value[nested]
    raise ValueError("Cannot locate draw list in API response; inspect API schema before changing parser")

def parse_draw(raw):
    if not isinstance(raw, dict): raise ValueError("draw row is not an object")
    issue = first(raw, ALIASES["draw_nbr"])
    code = first(raw, ALIASES["draw_number"])
    if issue is None or code is None: raise ValueError(f"missing issue/code fields: {list(raw)[:20]}")
    issue = int(re.sub(r"\D", "", str(issue)))
    digits = [int(x) for x in re.findall(r"\d+", str(code))]
    if len(digits) != 3 or any(x < 0 or x > 9 for x in digits):
        raise ValueError(f"invalid three-digit draw code for issue {issue}: {code!r}")
    total = sum(digits)
    size = "大" if total >= 14 else "小"
    parity = "双" if total % 2 == 0 else "单"
    combo = size + parity
    date = first(raw, ALIASES["draw_date"])
    clock = first(raw, ALIASES["draw_time"])
    open_time = first(raw, ("opentime","openTime","timestamp","open_timestamp"))
    if open_time and (not date or not clock):
        s = str(open_time).replace("T"," ")
        if len(s) >= 10: date = date or s[:10]
        if len(s) >= 16: clock = clock or s[11:19]
    date = str(date or "")[:10]
    clock = str(clock or "")[:8]
    return (issue, date, clock, "+".join(map(str,digits)), total, size, parity, combo)

def validate(row):
    issue, date, clock, code, total, size, parity, combo = row
    if issue <= 0: raise ValueError("issue must be positive")
    if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date): raise ValueError(f"bad date: {date}")
    if clock and not re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", clock): raise ValueError(f"bad time: {clock}")
    digits = [int(x) for x in code.split("+")]
    if len(digits) != 3 or sum(digits) != total: raise ValueError(f"sum mismatch at issue {issue}")
    if size != ("大" if total >= 14 else "小") or parity != ("双" if total%2==0 else "单") or combo != size+parity:
        raise ValueError(f"category mismatch at issue {issue}")

def init_db(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS draws(
      draw_nbr INTEGER PRIMARY KEY, draw_date TEXT NOT NULL, draw_time TEXT NOT NULL,
      draw_number TEXT NOT NULL, draw_num INTEGER NOT NULL, size_type TEXT NOT NULL,
      parity_type TEXT NOT NULL, combination_type TEXT NOT NULL, source TEXT NOT NULL,
      inserted_at TEXT NOT NULL)""")
    conn.execute("CREATE TABLE IF NOT EXISTS sync_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, ran_at TEXT, source TEXT, fetched INTEGER, inserted INTEGER, latest_issue INTEGER, gaps TEXT, status TEXT, detail TEXT)")
    conn.commit()

def import_csv(conn, path):
    if not path or not Path(path).exists(): return 0
    inserted = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing: raise ValueError(f"bootstrap CSV missing columns: {sorted(missing)}")
        for raw in reader:
            row = (int(raw["draw_nbr"]), raw["draw_date"], raw["draw_time"], raw["draw_number"], int(raw["draw_num"]), raw["size_type"], raw["parity_type"], raw["combination_type"])
            validate(row)
            cur = conn.execute("INSERT OR IGNORE INTO draws VALUES(?,?,?,?,?,?,?,?,?,?)",
                (*row, "bootstrap_csv", datetime.now(timezone.utc).isoformat()))
            inserted += cur.rowcount
    conn.commit()
    return inserted

def fetch_api():
    req = Request(API_URL, headers={"User-Agent":"PC28-DeepFusion-Engine/1.0","Accept":"application/json"})
    with urlopen(req, timeout=30) as response:
        if response.status != 200: raise RuntimeError(f"API HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8-sig"))
    rows = [parse_draw(x) for x in extract_rows(payload)]
    for row in rows: validate(row)
    return sorted({r[0]:r for r in rows}.values(), key=lambda r:r[0])

def write_exports(conn):
    DATA.mkdir(parents=True, exist_ok=True)
    rows = conn.execute("SELECT draw_nbr,draw_date,draw_time,draw_number,draw_num,size_type,parity_type,combination_type FROM draws ORDER BY draw_nbr").fetchall()
    for name, selected in (("raw_history.csv", rows), ("rolling_10000.csv", rows[-ROLLING_SIZE:])):
        with open(DATA/name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(FIELDS); w.writerows(selected)
    issues = [r[0] for r in rows]
    gaps = []
    for a,b in zip(issues, issues[1:]):
        if b-a > 1: gaps.append([a+1,b-1])
    latest = issues[-1] if issues else None
    manifest = {
      "updated_at": datetime.now(timezone.utc).isoformat(), "total_archived":len(rows),
      "rolling_size":min(len(rows),ROLLING_SIZE), "first_issue":issues[0] if issues else None,
      "latest_issue":latest, "gap_ranges":gaps, "api_url":API_URL,
      "note":"Raw archive is append-only; rolling file is the latest N draws."
    }
    (DATA/"dataset_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest, gaps

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    try:
        init_db(conn)
        boot = import_csv(conn, os.getenv("PC28_BOOTSTRAP_CSV", str(DATA/"bootstrap.csv")))
        fetched = fetch_api()
        before = conn.execute("SELECT COALESCE(MAX(draw_nbr),0) FROM draws").fetchone()[0]
        inserted = 0
        for row in fetched:
            cur = conn.execute("INSERT OR IGNORE INTO draws VALUES(?,?,?,?,?,?,?,?,?,?)",
                (*row, "api", datetime.now(timezone.utc).isoformat()))
            inserted += cur.rowcount
        conn.commit()
        manifest, gaps = write_exports(conn)
        detail = f"bootstrap_inserted={boot}; api_inserted={inserted}"
        status = "ok"
        print(json.dumps({"status":status,"fetched":len(fetched),"inserted":inserted,"bootstrap_inserted":boot,**manifest},ensure_ascii=False))
        conn.execute("INSERT INTO sync_runs(ran_at,source,fetched,inserted,latest_issue,gaps,status,detail) VALUES(?,?,?,?,?,?,?,?)",
          (datetime.now(timezone.utc).isoformat(),API_URL,len(fetched),inserted,manifest["latest_issue"],json.dumps(gaps),status,detail))
        conn.commit()
        if before and manifest["latest_issue"] and manifest["latest_issue"] > before + 1:
            print(f"WARNING: latest issue advanced from {before} to {manifest['latest_issue']}; check gap_ranges and backfill API coverage.",file=sys.stderr)
    except Exception as exc:
        conn.rollback()
        print(f"SYNC FAILED: {type(exc).__name__}: {exc}",file=sys.stderr)
        try:
            conn.execute("INSERT INTO sync_runs(ran_at,source,fetched,inserted,latest_issue,gaps,status,detail) VALUES(?,?,?,?,?,?,?,?)",
              (datetime.now(timezone.utc).isoformat(),API_URL,0,0,None,"[]","error",str(exc)))
            conn.commit()
        except Exception: pass
        raise
    finally:
        conn.close()

if __name__ == "__main__": main()
