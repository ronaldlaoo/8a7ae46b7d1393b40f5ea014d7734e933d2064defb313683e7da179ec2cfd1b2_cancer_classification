from datetime import datetime, timedelta
from pathlib import Path
import joblib
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from airflow import DAG
from airflow.operators.python import PythonOperator

from feature_engineering import add_features
from loguru import logger

# Set up logging
logger.remove()
logger.add(sys.stdout, level="DEBUG", enqueue=True, backtrace=True, diagnose=True)

# Paths
DATA_PATH = Path("/opt/airflow/data/raw/cancer_dataset.csv")
SPLIT_PATH = Path("/opt/airflow/work/split.joblib")
MODEL_PATH = Path("/opt/airflow/models/model.pkl")
REPORT_PATH = Path("/opt/airflow/reports/metrics.txt")


def t_preprocess():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    logger.info("Reading dataset from {}", DATA_PATH)
    df = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"])
    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), SPLIT_PATH)
    logger.info("Saved split to {}", SPLIT_PATH)


def t_train_model():
    X_train, _, y_train, _ = joblib.load(SPLIT_PATH)
    X_train = add_features(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info("Model trained and saved to {}", MODEL_PATH)


def t_evaluate_model():
    _, X_test, _, y_test = joblib.load(SPLIT_PATH)
    X_test = add_features(X_test)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info("Model accuracy: {:.2f}%", accuracy * 100)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")

    logger.info("Saved evaluation report to {}", REPORT_PATH)


with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2025, 8, 1),
    schedule="@once",
    catchup=False,
    default_args={"retries": 3, "retry_delay": timedelta(minutes=1)},
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=t_preprocess,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=t_train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=t_evaluate_model,
    )

    preprocess >> train >> evaluate
