"""Validate frozen 20-draw plans against actual results without editing plans."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from features.engine import CLASS_TO_IDX, load_and_prepare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rolling_10000.csv"
PLAN_DIR = ROOT / "prediction" / "plans"
VALIDATION_DIR = ROOT / "prediction" / "validation"


def canonical_hash(record: dict) -> str:
    payload = {key: value for key, value in record.items() if key != "sha256"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_all() -> list[dict]:
    df = load_and_prepare(DATA)
    actual_by_issue = {
        int(row.draw_nbr): str(row.combination_type)
        for row in df[["draw_nbr", "combination_type"]].itertuples(index=False)
    }
    latest_issue = max(actual_by_issue) if actual_by_issue else 0
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []

    for plan_path in sorted(PLAN_DIR.glob("*.json")):
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_id = str(plan.get("plan_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", plan_id):
            raise ValueError(f"Unsafe or missing plan_id in {plan_path}")
        if plan.get("frozen") is not True:
            raise ValueError(f"Plan is not frozen: {plan_path}")
        expected_hash = canonical_hash(plan)
        if plan.get("sha256") != expected_hash:
            raise ValueError(f"Frozen plan hash mismatch: {plan_path}")

        predictions = plan.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != 20:
            raise ValueError(f"Plan must contain exactly 20 predictions: {plan_path}")

        results = []
        for item in predictions:
            issue = int(item["issue"])
            if issue not in actual_by_issue:
                continue
            predicted = str(item["prediction"])
            actual = actual_by_issue[issue]
            if predicted not in CLASS_TO_IDX or actual not in CLASS_TO_IDX:
                raise ValueError(f"Unknown class in plan/result at issue {issue}")
            results.append({
                "issue": issue,
                "prediction": predicted,
                "actual": actual,
                "hit": predicted == actual,
            })

        output_path = VALIDATION_DIR / f"{plan_id}.json"
        previous = None
        if output_path.exists():
            previous = json.loads(output_path.read_text(encoding="utf-8"))
            old_results = previous.get("results", [])
            if results[:len(old_results)] != old_results:
                raise ValueError(
                    f"Previously validated results changed for {plan_id}; "
                    "refusing to rewrite the audit trail."
                )

        if not results:
            summaries.append({"plan_id": plan_id, "validated": 0, "completed": False})
            continue

        hits = sum(1 for row in results if row["hit"])
        report = {
            "plan_id": plan_id,
            "plan_sha256": expected_hash,
            "model_version": plan["model_version"],
            "last_historical_issue": int(plan["last_historical_issue"]),
            "latest_available_issue": latest_issue,
            "validated_periods": len(results),
            "hits": hits,
            "hit_rate": round(hits / len(results), 6),
            "completed": len(results) == 20,
            "results": results,
        }
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if previous != report:
            output_path.write_text(encoded, encoding="utf-8")
        summaries.append({
            "plan_id": plan_id,
            "validated": len(results),
            "hits": hits,
            "completed": report["completed"],
            "report": str(output_path.relative_to(ROOT)),
        })

    print(json.dumps({
        "plans_found": len(list(PLAN_DIR.glob("*.json"))),
        "latest_available_issue": latest_issue,
        "summaries": summaries,
    }, ensure_ascii=False, indent=2))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    validate_all()
