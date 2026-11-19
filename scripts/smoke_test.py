"""End-to-end smoke checks for a running synthetic demo stack."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Request to {url} failed ({exc.code}): {body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = request_json(f"{base}/health")
    assert health["status"] == "ok", health
    assert health["database_ready"] is True
    assert health["model_registry_ready"] is True

    portfolio = request_json(f"{base}/api/v1/portfolio/summary")
    assert portfolio["applications"] > 0
    assert 0 <= portfolio["default_rate"] <= 1

    registry = request_json(f"{base}/api/v1/model/registry")
    assert registry["active_version"] == health["model_version"]
    assert registry["versions"]
    active_versions = [item for item in registry["versions"] if item["status"] == "active"]
    assert len(active_versions) == 1
    assert len(active_versions[0]["artifact_sha256"]) == 64
    assert len(active_versions[0]["dataset_sha256"]) == 64

    evaluation = request_json(f"{base}/api/v1/model/evaluation")
    assert evaluation["final_test"]["sample_size"] > 0
    assert "subgroup_diagnostics" in evaluation
    assert evaluation["calibration_diagnostics"]["bins"]
    assert evaluation["final_test_confidence_intervals"]["roc_auc"]["lower_95"] <= evaluation["final_test_confidence_intervals"]["roc_auc"]["upper_95"]

    sample = request_json(f"{base}/api/v1/assessments/sample")
    features = sample["features"]
    scored = request_json(
        f"{base}/api/v1/assessments/explain", {"features": features}
    )
    assert 0 <= scored["probability_of_default"] <= 1
    assert "risk_increasing" in scored["explanation"]

    batch = request_json(
        f"{base}/api/v1/assessments/batch", {"applications": [features]}
    )
    assert batch["submitted"] == 1
    assert batch["scored"] == 1
    assert batch["rejected"] == 0

    audit = request_json(f"{base}/api/v1/audit/summary")
    assert audit["assessments_recorded"] >= 2
    assert "applicant_features" not in audit["stored_fields"]

    print(
        json.dumps(
            {
                "status": "passed",
                "portfolio_applications": portfolio["applications"],
                "held_out_test_size": evaluation["final_test"]["sample_size"],
                "registered_model_versions": len(registry["versions"]),
                "single_assessment": scored["risk_band"],
                "batch_scored": batch["scored"],
                "audit_records": audit["assessments_recorded"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
