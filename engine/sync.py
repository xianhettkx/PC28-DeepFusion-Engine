#!/usr/bin/env python3
"""PC28 append-only history synchronizer with strict continuity checks.

The API returns at most 100 recent draws. The synchronizer only accepts a run
when every issue between the stored archive and the API's latest issue exists.
It never silently reports success while gaps remain.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "pc28.sqlite3"
API_URL = os.getenv("PC28_API_URL", "https://pc28.help/api/kj.json?nbr=100")
ROLLING_SIZE = int(os.getenv("PC28_ROLLING_SIZE", "10000"))
FIELDS = [
    "draw_nbr", "draw_date", "draw_time", "draw_number", "draw_num",
    "size_type", "parity_type", "combination_type",
]
ALIASES = {
    "draw_nbr": ("draw_nbr", "nbr", "expect", "issue", "period", "qihao"),
    "draw_date": ("draw_date", "date", "open_date"),
    "draw_time": ("draw_time", "time", "open_time"),
    "draw_number": ("draw_number", "opencode", "open_code", "code", "result", "number"),
}


def first(obj: dict, names: tuple[str, ...]):
    for name in names:
        value = obj.get(name)
        if value not in (None, ""):
            return value
    return None


def extract_rows(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("API JSON root must be an object or array")
    for key in ("data", "list", "rows", "result", "records", "history"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for nested in ("data", "list", "rows", "records"):
                if isinstance(value.get(nested), list):
                    return value[nested]
    raise ValueError("API response contains no recognized draw list")


def parse_draw(raw: dict) -> tuple:
    if not isinstance(raw, dict):
        raise ValueError("draw row is not an object")
    issue_raw = first(raw, ALIASES["draw_nbr"])
    code_raw = first(raw, ALIASES["draw_number"])
    if issue_raw is None or code_raw is None:
        raise ValueError(f"missing issue/code fields: {list(raw)[:20]}")
    issue_text = str(issue_raw).strip()
    if not re.fullmatch(r"\d+", issue_text):
        raise ValueError(f"invalid issue number: {issue_raw!r}")
    issue = int(issue_text)

    code_text = str(code_raw).strip().split("=", 1)[0]
    match = re.fullmatch(r"\s*(\d)\s*\+\s*(\d)\s*\+\s*(\d)\s*", code_text)
    if not match:
        raise ValueError(f"invalid three-ball result for issue {issue}: {code_raw!r}")
    digits = [int(x) for x in match.groups()]
    total = sum(digits)
    size = "大" if total >= 14 else "小"
    parity = "双" if total % 2 == 0 else "单"
    combo = size + parity

    date = str(first(raw, ALIASES["draw_date"]) or "")[:10]
    clock = str(first(raw, ALIASES["draw_time"]) or "")[:8]
    return (issue, date, clock, "+".join(map(str, digits)), total, size, parity, combo)


def validate(row: tuple) -> None:
    issue, date, clock, code, total, size, parity, combo = row
    if issue <= 0:
        raise ValueError("issue must be positive")
    if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"bad date at issue {issue}: {date}")
    if clock and not re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", clock):
        raise ValueError(f"bad time at issue {issue}: {clock}")
    digits = [int(x) for x in code.split("+")]
    if len(digits) != 3 or any(x < 0 or x > 9 for x in digits) or sum(digits) != total:
        raise ValueError(f"invalid number/sum at issue {issue}")
    expected_size = "大" if total >= 14 else "小"
    expected_parity = "双" if total % 2 == 0 else "单"
    if (size, parity, combo) != (expected_size, expected_parity, expected_size + expected_parity):
        raise ValueError(f"category mismatch at issue {issue}")


def result_signature(row: tuple) -> tuple:
    # Compare the actual draw result, not formatting of timestamps.
    return row[3], row[4], row[5], row[6], row[7]


def deduplicate(rows: list[tuple], source: str) -> list[tuple]:
    by_issue = {}
    for row in rows:
        validate(row)
        issue = row[0]
        previous = by_issue.get(issue)
        if previous is not None and result_signature(previous) != result_signature(row):
            raise ValueError(f"{source} has conflicting results for issue {issue}")
        by_issue[issue] = row
    return [by_issue[k] for k in sorted(by_issue)]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS draws(
      draw_nbr INTEGER PRIMARY KEY, draw_date TEXT NOT NULL, draw_time TEXT NOT NULL,
      draw_number TEXT NOT NULL, draw_num INTEGER NOT NULL, size_type TEXT NOT NULL,
      parity_type TEXT NOT NULL, combination_type TEXT NOT NULL, source TEXT NOT NULL,
      inserted_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sync_runs(
      id INTEGER PRIMARY KEY AUTOINCREMENT, ran_at TEXT, source TEXT, fetched INTEGER,
      inserted INTEGER, latest_issue INTEGER, gaps TEXT, status TEXT, detail TEXT)""")
    conn.commit()


def insert_or_verify(conn: sqlite3.Connection, row: tuple, source: str) -> int:
    issue = row[0]
    existing = conn.execute(
        "SELECT draw_nbr,draw_date,draw_time,draw_number,draw_num,size_type,parity_type,combination_type "
        "FROM draws WHERE draw_nbr=?", (issue,)
    ).fetchone()
    if existing:
        if result_signature(tuple(existing)) != result_signature(row):
            raise ValueError(f"conflicting draw result for existing issue {issue}")
        return 0
    conn.execute(
        "INSERT INTO draws VALUES(?,?,?,?,?,?,?,?,?,?)",
        (*row, source, datetime.now(timezone.utc).isoformat()),
    )
    return 1


def import_csv(conn: sqlite3.Connection, path: Path, source: str) -> int:
    if not path.exists():
        return 0
    inserted = 0
    seen = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        for raw in reader:
            row = (
                int(raw["draw_nbr"]), raw["draw_date"].strip(), raw["draw_time"].strip(),
                raw["draw_number"].strip(), int(raw["draw_num"]), raw["size_type"].strip(),
                raw["parity_type"].strip(), raw["combination_type"].strip(),
            )
            validate(row)
            if row[0] in seen and result_signature(seen[row[0]]) != result_signature(row):
                raise ValueError(f"{path.name} has conflicting duplicate issue {row[0]}")
            seen[row[0]] = row
            inserted += insert_or_verify(conn, row, source)
    conn.commit()
    return inserted


def fetch_api() -> list[tuple]:
    req = Request(API_URL, headers={
        "User-Agent": "PC28-DeepFusion-Engine/2.0",
        "Accept": "application/json",
    })
    with urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"API HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8-sig"))
    if isinstance(payload, dict) and payload.get("message") not in (None, "", "success"):
        raise RuntimeError(f"API returned error: {payload.get('message')}")
    rows = deduplicate([parse_draw(item) for item in extract_rows(payload)], "API")
    if not rows:
        raise RuntimeError("API returned zero valid draws")
    if len(rows) > 100:
        raise RuntimeError(f"API returned {len(rows)} rows; expected at most 100")
    return rows


def find_gaps(issues: list[int]) -> list[list[int]]:
    return [[left + 1, right - 1] for left, right in zip(issues, issues[1:]) if right - left > 1]


def write_exports(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT draw_nbr,draw_date,draw_time,draw_number,draw_num,size_type,parity_type,combination_type "
        "FROM draws ORDER BY draw_nbr"
    ).fetchall()
    if not rows:
        raise RuntimeError("No history records exist")
    issues = [int(row[0]) for row in rows]
    gaps = find_gaps(issues)
    if gaps:
        preview = ", ".join(f"{a}-{b}" for a, b in gaps[:10])
        raise RuntimeError(f"history is not continuous; missing issue range(s): {preview}")
    DATA.mkdir(parents=True, exist_ok=True)
    for name, selected in (("raw_history.csv", rows), ("rolling_10000.csv", rows[-ROLLING_SIZE:])):
        with (DATA / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerows(selected)
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_archived": len(rows),
        "rolling_size": min(len(rows), ROLLING_SIZE),
        "first_issue": issues[0],
        "latest_issue": issues[-1],
        "duplicate_issues": 0,
        "gap_ranges": [],
        "continuous": True,
        "api_url": API_URL,
        "note": "raw_history.csv is the complete archive; rolling_10000.csv is the latest 10000 draws.",
    }
    (DATA / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    try:
        init_db(conn)
        # On GitHub's fresh runner, restore the committed full archive first.
        archive = DATA / "raw_history.csv"
        bootstrap = DATA / "bootstrap.csv"
        source_path = archive if archive.exists() else bootstrap
        if not source_path.exists():
            raise FileNotFoundError("Neither data/raw_history.csv nor data/bootstrap.csv exists")
        imported = import_csv(conn, source_path, "raw_history_csv" if source_path == archive else "bootstrap_csv")

        api_rows = fetch_api()
        old_latest = int(conn.execute("SELECT COALESCE(MAX(draw_nbr),0) FROM draws").fetchone()[0])
        api_latest = api_rows[-1][0]
        if api_latest < old_latest:
            raise RuntimeError(
                f"API is stale: API latest={api_latest}, stored latest={old_latest}; archive not changed"
            )

        api_issues = {row[0] for row in api_rows}
        if api_latest > old_latest:
            missing_from_api = [
                issue for issue in range(old_latest + 1, api_latest + 1)
                if issue not in api_issues
            ]
            if missing_from_api:
                raise RuntimeError(
                    "API's latest-100 window cannot bridge the archive: "
                    f"missing {missing_from_api[0]}..{missing_from_api[-1]} "
                    f"({len(missing_from_api)} issue(s)); refusing partial sync"
                )

        inserted = 0
        for row in api_rows:
            inserted += insert_or_verify(conn, row, "api")
        conn.commit()

        manifest = write_exports(conn)
        detail = f"source={source_path.name}; imported={imported}; api_fetched={len(api_rows)}; api_inserted={inserted}"
        conn.execute(
            "INSERT INTO sync_runs(ran_at,source,fetched,inserted,latest_issue,gaps,status,detail) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), API_URL, len(api_rows), inserted,
             manifest["latest_issue"], "[]", "ok", detail),
        )
        conn.commit()
        print(json.dumps({
            "status": "ok", "imported": imported, "api_fetched": len(api_rows),
            "api_inserted": inserted, **manifest,
        }, ensure_ascii=False))
    except Exception as exc:
        conn.rollback()
        print(f"SYNC FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            conn.execute(
                "INSERT INTO sync_runs(ran_at,source,fetched,inserted,latest_issue,gaps,status,detail) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), API_URL, 0, 0, None, "[]", "error", str(exc)),
            )
            conn.commit()
        except Exception:
            pass
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
