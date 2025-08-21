# deploy/airflow/dags/ml_pipeline_dag.py
from datetime import datetime, timedelta
from pathlib import Path
import json
import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator


from src.data_preprocessing import preprocess_data
from src.feature_engineering import add_features
from src.model_training import ml_train_model
from src.evaluation import ml_evaluate_model
from src.drift_detection import detect_drift


mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("cancer_pipeline_airflow")

# Paths
REPORTS_PATH = Path("reports") / "drift_report.json"


def task_preprocess():
    return preprocess_data()


def task_feature_engineering(ti):
    X_train, X_test, y_train, y_test, *_ = ti.xcom_pull(task_ids="preprocess_data")
    return add_features(X_train).to_dict(), add_features(X_test).to_dict(), y_train.tolist(), y_test.tolist()


def task_train(ti):
    X_train_dict, X_test_dict, y_train, y_test = ti.xcom_pull(task_ids="feature_engineering")
    import pandas as pd
    X_train = pd.DataFrame(X_train_dict)
    model = ml_train_model(X_train, y_train)
    return model


def task_evaluate(ti):
    model = ti.xcom_pull(task_ids="train_model")
    _, X_test_dict, _, y_test = ti.xcom_pull(task_ids="feature_engineering")
    import pandas as pd
    X_test = pd.DataFrame(X_test_dict)
    return ml_evaluate_model(model, X_test, y_test)


def task_drift():
    return detect_drift("data/test.csv", "data/drifted_test.csv")


def branch_on_drift():
    with open(REPORTS_PATH, "r") as f:
        drift_result = json.load(f)
    return "retrain_model" if drift_result["drift_detected"] else "pipeline_complete"


def task_retrain():
    from pathlib import Path
    import pandas as pd

    root = Path(__file__).resolve().parents[2]  
    train = pd.read_csv(root / "data" / "train.csv")
    X = train.drop(columns=["target"])
    y = train["target"]
    return ml_train_model(X, y)


with DAG(
    dag_id="ml_pipeline_dag",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    description="ML pipeline with drift detection and retraining",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess,
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_feature_engineering,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=task_train,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate,
    )

    drift_task = PythonOperator(
        task_id="drift_detection",
        python_callable=task_drift,
    )

    branch_task = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=branch_on_drift,
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=task_retrain,
    )

    complete_task = EmptyOperator(task_id="pipeline_complete")

    # Dependencies
    preprocess_task >> feature_engineering_task >> train_task >> evaluate_task
    evaluate_task >> drift_task >> branch_task
    branch_task >> [retrain_task, complete_task]
