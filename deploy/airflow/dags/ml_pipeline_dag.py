from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data_preprocessing import load_dataset
from src.feature_engineering import add_features
from src.model_training import train_model
from src.evaluation import evaluate_model

def run_preprocessing(**kwargs):
    X_train, X_test, y_train, y_test = load_dataset()
    # Push the data split into XCom if you need to pass it to the next tasks.
    # For simplicity, we'll assume you have an external storage (e.g., files) or database for intermediate outputs.
    return X_train, X_test, y_train, y_test

def run_feature_engineering(ti, **kwargs):
    # Retrieve data from previous task, if using XCom
    # For example, load data from file or XCom (not recommended for large data, so persist if needed)
    X_train, X_test, _, _ = ti.xcom_pull(task_ids='preprocess')
    X_train = add_features(X_train)
    X_test = add_features(X_test)
    return X_train, X_test

def run_training(ti, **kwargs):
    X_train, _, y_train, _ = ti.xcom_pull(task_ids='preprocess')
    # Optionally, combine with results of feature engineering if in different storage
    train_model(X_train, y_train)

def run_evaluation(ti, **kwargs):
    # Assume the trained model is saved in "models/model.pkl"
    _, X_test, _, y_test = ti.xcom_pull(task_ids='preprocess')
    model_path = "models/model.pkl"
    evaluate_model(model_path, X_test, y_test)

default_args = {
    "retries": 1,
}

with DAG(
    dag_id="ml_pipeline_dag",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=run_preprocessing,
        provide_context=True,  # if you use XComs, etc.
    )

    feature_eng = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_feature_engineering,
        provide_context=True,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
        provide_context=True,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluation,
        provide_context=True,
    )

    preprocess >> feature_eng >> train >> evaluate
