import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset  
from pathlib import Path
from typing import Dict, Any
import json
from loguru import logger

# def get_root():
#    root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
#    return root

import os

def get_root():
    # Prefer env var AIRFLOW_HOME or project root inside container
    if "AIRFLOW_HOME" in os.environ:
        return Path(os.environ["AIRFLOW_HOME"])
    # fallback: mounted /opt/airflow path
    airflow_path = Path("/opt/airflow")
    if airflow_path.exists():
        return airflow_path
    # fallback to git detection (for local dev)
    return next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())


def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    root = get_root()
    ref_df = pd.read_csv(root / reference_data_path).drop(columns=["target"])
    cur_df = pd.read_csv(root / current_data_path).drop(columns=["target"])

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df, current_data=cur_df)
    result = snapshot.dict()
    logger.info("Drift detection report generated")
    drift_detected = False
    feature_drifts = {}

    for metric in result["metrics"]:
        if metric["metric_id"].startswith("DriftedColumnsCount"):
            # Dataset-level drift
            drift_detected = metric["value"]["share"] > 0.5  # Evidently’s 50% rule
        elif metric["metric_id"].startswith("ValueDrift"):
            # Feature-level drift
            col_name = metric["metric_id"].split("column=")[-1].strip(")")
            feature_drifts[col_name] = float(metric["value"])

    # Overall drift score (average of feature drift scores)
    overall_drift_score = (
        float(sum(feature_drifts.values()) / len(feature_drifts))
        if feature_drifts else 0.0
    )

    payload: Dict[str, Any] = {
        "drift_detected": drift_detected,
        "feature_drifts": feature_drifts,
        "overall_drift_score": overall_drift_score,
    }

    reports_path = root / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)
    with (reports_path / "drift_report.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Drift report saved to {reports_path / 'drift_report.json'}")
    return payload
