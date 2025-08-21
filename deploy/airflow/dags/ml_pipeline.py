from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import mlflow
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle

from loguru import logger

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

sys.path.append("/opt/airflow/src")

from src.data_preprocessing import preprocess_data
from src.feature_engineering import add_features
# from src.model_training import ml_train_model
from src.evaluation import ml_evaluate_model
from src.drift_detection import detect_drift

# MLflow setup
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("cancer_pipeline_airflow")

# Paths
ROOT = Path("/opt/airflow")
DATA_PATH = ROOT / "data"
REPORTS_PATH = ROOT / "reports" / "drift_report.json"


# --- Tasks ---
def task_preprocess():
    logger.info(">>> Starting preprocessing task")
    preprocess_data()
    logger.info(">>> Preprocessing completed. Train/test + drifted CSVs saved.")
    return "preprocessing done"


def task_feature_engineering():
    logger.info(">>> Starting feature engineering")
    train = pd.read_csv(DATA_PATH / "train.csv")
    test = pd.read_csv(DATA_PATH / "test.csv")

    X_train = add_features(train.drop(columns=["target"]))
    X_test = add_features(test.drop(columns=["target"]))

    X_train.assign(target=train["target"]).to_csv(DATA_PATH / "train_fe.csv", index=False)
    X_test.assign(target=test["target"]).to_csv(DATA_PATH / "test_fe.csv", index=False)

    logger.info(">>> Feature engineering completed. train_fe.csv and test_fe.csv saved.")
    return "feature engineering done"



def task_train():
    logger.info(">>> Starting model training with GridSearchCV")
    train = pd.read_csv(DATA_PATH / "train_fe.csv")
    X, y = train.drop(columns=["target"]), train["target"]

    # Parameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "random_state": [1, 42],
    }

    # Grid search
    grid = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    model = grid.best_estimator_

    logger.info(f">>> Best params: {grid.best_params_}")

    # Save best model
    artifacts_path = ROOT / "mlflow" / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    model_pickle_path = artifacts_path / "model.pkl"
    with open(model_pickle_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f">>> Training completed. Best model saved to {model_pickle_path}")
    return "model trained"

def task_evaluate():
    logger.info(">>> Starting evaluation")
    test = pd.read_csv(DATA_PATH / "test_fe.csv")
    X, y = test.drop(columns=["target"]), test["target"]

    model_path = ROOT / "mlflow" / "artifacts" / "model.pkl"
    model = joblib.load(model_path)

    ml_evaluate_model(model, X, y)
    logger.info(">>> Evaluation completed. Metrics logged to MLflow and JSON report generated.")
    return "evaluation done"


def task_drift():
    logger.info(">>> Running drift detection")
    detect_drift("data/test.csv", "data/drifted_test.csv")

    with open(REPORTS_PATH, "r") as f:
        drift_result = json.load(f)

    logger.info(f">>> Drift detection report: {drift_result}")
    return "drift detection done"


def branch_on_drift():
    logger.info(">>> Branching on drift detection results")
    with open(REPORTS_PATH, "r") as f:
        drift_result = json.load(f)

    if drift_result.get("drift_detected"):
        logger.info(">>> Drift detected. Branching to retraining.")
        return "retrain_model"
    else:
        logger.info(">>> No significant drift. Branching to pipeline_complete.")
        return "pipeline_complete"


def task_retrain():
    logger.info(">>> Starting retraining with GridSearchCV on drifted data")
    drifted_train = pd.read_csv(DATA_PATH / "drifted_train.csv")

    # Apply feature engineering (same as normal training)
    X = add_features(drifted_train.drop(columns=["target"]))
    y = drifted_train["target"]

    # Parameter grid (same as train)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "random_state": [1, 42],
    }

    # Grid search
    grid = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    model = grid.best_estimator_

    logger.info(f">>> Best params (retrain): {grid.best_params_}")

    # Overwrite saved model
    artifacts_path = ROOT / "mlflow" / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    model_pickle_path = artifacts_path / "model.pkl"
    with open(model_pickle_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f">>> Retraining completed. New model saved to {model_pickle_path}")
    return "retrained"


# --- DAG Definition ---
with DAG(
    dag_id="ml_pipeline_dag",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    description="ML pipeline with drift detection and retraining",
    schedule="@once",
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    preprocess_task = PythonOperator(task_id="preprocess_data", python_callable=task_preprocess)
    feature_engineering_task = PythonOperator(task_id="feature_engineering", python_callable=task_feature_engineering)
    train_task = PythonOperator(task_id="train_model", python_callable=task_train)
    evaluate_task = PythonOperator(task_id="evaluate_model", python_callable=task_evaluate)
    drift_task = PythonOperator(task_id="drift_detection", python_callable=task_drift)
    branch_task = BranchPythonOperator(task_id="branch_on_drift", python_callable=branch_on_drift)
    retrain_task = PythonOperator(task_id="retrain_model", python_callable=task_retrain)
    complete_task = EmptyOperator(task_id="pipeline_complete")

    # Dependencies
    preprocess_task >> feature_engineering_task >> train_task >> evaluate_task
    evaluate_task >> drift_task >> branch_task
    branch_task >> [retrain_task, complete_task]
