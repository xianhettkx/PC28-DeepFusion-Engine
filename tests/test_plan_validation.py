"""Tests for immutable forecast plan hash verification."""
from prediction.validate_plans import canonical_hash


def test_plan_hash_ignores_digest_field_only():
    plan = {
        "plan_id": "DF-test",
        "frozen": True,
        "predictions": [{"issue": 123, "prediction": "大单"}],
    }
    digest = canonical_hash(plan)
    plan["sha256"] = digest
    assert canonical_hash(plan) == digest


def test_plan_hash_detects_forecast_tampering():
    plan = {
        "plan_id": "DF-test",
        "frozen": True,
        "predictions": [{"issue": 123, "prediction": "大单"}],
    }
    digest = canonical_hash(plan)
    plan["predictions"][0]["prediction"] = "小双"
    assert canonical_hash(plan) != digest
