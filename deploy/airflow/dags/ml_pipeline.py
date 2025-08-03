import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from airflow import DAG
from airflow.operators.python import PythonOperator


# Logger setup
from loguru import logger
logger.remove()
logger.add(sys.stdout, level="DEBUG", enqueue=True, backtrace=True, diagnose=True)

DATA_PATH = Path("/opt/airflow/data/raw/cancer_dataset.csv")
MODEL_PATH = Path("/opt/airflow/models")
WORK_PATH = Path("/opt/airflow/work")

def t_preprocess():
    logger.info("Reading from: {}", DATA_PATH)
    df = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"])
    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    split_path = WORK_PATH / "split.joblib"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), split_path)
    logger.info("Saved split to {}", split_path)
    return {"split_path": str(split_path)}

with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2025, 8, 1),
    schedule="@once",
    catchup=False,
    default_args={"retries": 3, "retry_delay":timedelta(minutes=1)},
) as dag:
    preprocess = PythonOperator(task_id="preprocess", python_callable=t_preprocess)
