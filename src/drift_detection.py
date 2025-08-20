import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path
from typing import Dict, Any
import json

def get_root():
   root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
   return root

def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    """
    Detects data drift between reference and current datasets.

    Args:
        reference_data_path: Path (relative to root) to reference CSV
        current_data_path: Path (relative to root) to current CSV

    Returns:
        Dictionary with keys: 'drift_detected', 'feature_drifts', 'overall_drift_score'
    """
    root = get_root()
    ref_path = root / reference_data_path
    ref_df = pd.read_csv(ref_path).drop(columns=["target"])

    cur_path = root / current_data_path
    cur_df = pd.read_csv(cur_path).drop(columns=["target"])

    # Drift Analysis
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    result = report.as_dict()

    # Extract dataset-level drift
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    # Extract feature-level drift scores
    drift_by_columns = result["metrics"][1]["result"]["drift_by_columns"]

    # Select at least 3 features 
    selected_features = (
        list(ref_df.columns[:3]) if ref_df.shape[1] >= 3 else list(ref_df.columns)
    )

    feature_drifts = {
        feature: float(drift_by_columns[feature]["drift_score"])
        for feature in selected_features
        if feature in drift_by_columns
    }

    # Overall drift score 
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

    return payload