from datetime import datetime
import os
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.task_group import TaskGroup

import custom_functions


default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1)
}


with DAG(dag_id="Daily_img_transfer", default_args=default_args, schedule_interval="@daily", catchup=False) as dag:
    start_dag = DummyOperator(task_id="start_dag")

    extract_images = PythonOperator(
        task_id="daily_extraction",
        python_callable=custom_functions.daily_extraction
    )

    load_images = PythonOperator(
        task_id="daily_load",
        python_callable=custom_functions.daily_load
    )

    clean_directory = PythonOperator(
        task_id="daily_clean",
        python_callable=custom_functions.daily_clean
    )
                               
    end_dag = DummyOperator(task_id="end_dag")

    start_dag >>  extract_images >> load_images >> clean_directory >> end_dag

with DAG(dag_id="Weekly_training", default_args=default_args, schedule_interval="@weekly", catchup=False) as dag2:
    start_dag = DummyOperator(task_id="start_dag")

    with TaskGroup(group_id="prepare_data") as prepare_data:
        weekly_extract = PythonOperator(
            task_id="weekly_extract",
            python_callable=custom_functions.weekly_s3_extract
        )

        organize_data = PythonOperator(
            task_id="organize_data",
            python_callable=custom_functions.organize_data
        )

        [weekly_extract >> organize_data]

    with TaskGroup(group_id="training") as new_training:
        check_mlflow = PythonOperator(
            task_id="check_mlflow",
            python_callable=custom_functions.check_mlflow
        )

        training = PythonOperator(
            task_id="train",
            python_callable=custom_functions.new_training
        )

        [check_mlflow >> training] # >> training

    with TaskGroup(group_id="monitoring") as monitoring:
        drift_detection = PythonOperator(
            task_id="drift_detection",
            python_callable=custom_functions.drift_detection
        )

        [drift_detection]

    end_dag = DummyOperator(task_id="end_dag")

    start_dag >> prepare_data >> new_training >> monitoring >> end_dag