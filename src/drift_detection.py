import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset  
from pathlib import Path
from typing import Dict, Any
import json
from loguru import logger

def get_root():
   root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
   return root

def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    root = get_root()
    ref_df = pd.read_csv(root / reference_data_path).drop(columns=["target"])
    cur_df = pd.read_csv(root / current_data_path).drop(columns=["target"])

    # Drift Analysis
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df, current_data=cur_df)
    result = snapshot.dict()
    logger.info("Drift detection completed")
    # Each metric entry has "result" key
    metric_result = result["metrics"][0].get("result", {})

    # Dataset-level drift
    drift_detected = metric_result.get("dataset_drift", False)

    # Feature-level drift
    drift_by_columns = metric_result.get("drift_by_columns", {})

    # Pick a few features
    selected_features = (
        list(ref_df.columns[:3]) if ref_df.shape[1] >= 3 else list(ref_df.columns)
    )
    feature_drifts = {
        feature: float(drift_by_columns[feature]["drift_score"])
        for feature in selected_features
        if feature in drift_by_columns
    }

    # Overall score
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
        
    logger.info("Drift report saved to reports/drift_report.json")
    return payload