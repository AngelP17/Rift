from __future__ import annotations

from datetime import datetime
import subprocess


try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except Exception:  # pragma: no cover - airflow is optional in tests/runtime
    DAG = None
    PythonOperator = None
    dag = None
else:
    PROJECT_ROOT = "/opt/airflow/project"

    def run_cli(args: list[str]) -> None:
        subprocess.run(["python3", "-m", "rift.cli.main", *args], cwd=PROJECT_ROOT, check=True)


    default_args = {
        "owner": "rift",
        "start_date": datetime(2026, 3, 9),
    }

    with DAG(
        dag_id="rift_fraud_pipeline",
        default_args=default_args,
        description="Local zero-cost Rift pipeline with ETL, training, fairness, and lakehouse materialization",
        schedule="@daily",
        catchup=False,
        tags=["rift", "fraud", "mlops"],
    ) as dag:
        generate_data = PythonOperator(
            task_id="generate_data",
            python_callable=lambda: run_cli(
                ["generate", "--txns", "10000", "--users", "1000", "--merchants", "200", "--fraud-rate", "0.02"]
            ),
        )

        train_model = PythonOperator(
            task_id="train_model",
            python_callable=lambda: run_cli(["train", "--model", "graphsage_xgb", "--time-split"]),
        )

        run_fairness = PythonOperator(
            task_id="run_fairness_audit",
            python_callable=lambda: run_cli(["fairness", "audit", "--sensitive-column", "channel"]),
        )

        build_lakehouse = PythonOperator(
            task_id="build_lakehouse_views",
            python_callable=lambda: run_cli(["lakehouse", "build"]),
        )

        generate_data >> train_model >> run_fairness >> build_lakehouse
