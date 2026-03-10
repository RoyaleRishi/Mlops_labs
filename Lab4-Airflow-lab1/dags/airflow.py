from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.lab import (
    build_dashboard,
    load_data,
    prepare_features,
    score_customer_profile,
    train_segment_model,
)

default_args = {
    "owner": "Rishi",
    "start_date": datetime(2026, 3, 10),
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="customer_segmentation_dashboard_lab",
    default_args=default_args,
    description="Custom Airflow lab for Gaussian mixture customer segmentation and dashboard publishing.",
    schedule=None,
    catchup=False,
    tags=["lab4", "segmentation", "dashboard"],
) as dag:
    load_customer_data = PythonOperator(
        task_id="load_customer_data",
        python_callable=load_data,
    )

    prepare_customer_features = PythonOperator(
        task_id="prepare_customer_features",
        python_callable=prepare_features,
        op_args=[load_customer_data.output],
    )

    train_segmentation_model = PythonOperator(
        task_id="train_segmentation_model",
        python_callable=train_segment_model,
        op_args=[prepare_customer_features.output, "customer_segmentation_model.pkl"],
    )

    score_custom_profile = PythonOperator(
        task_id="score_custom_profile",
        python_callable=score_customer_profile,
        op_args=[train_segmentation_model.output],
    )

    publish_dashboard = PythonOperator(
        task_id="publish_dashboard",
        python_callable=build_dashboard,
        op_args=[train_segmentation_model.output, score_custom_profile.output],
    )

    (
        load_customer_data
        >> prepare_customer_features
        >> train_segmentation_model
        >> score_custom_profile
        >> publish_dashboard
    )

if __name__ == "__main__":
    dag.test()
