from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_hello():
    print("Hello, Airflow!")

with DAG(
    dag_id='sample_working_dag',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    task1 = PythonOperator(
        task_id='print_hello',
        python_callable=print_hello
    )
