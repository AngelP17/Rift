from __future__ import annotations

from datetime import datetime
import subprocess


try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except Exception:  # pragma: no cover
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
        dag_id="rift_drift_monitoring",
        default_args=default_args,
        description="Local drift detection and retraining trigger workflow",
        schedule="@daily",
        catchup=False,
        tags=["rift", "drift", "governance"],
    ) as dag:
        check_drift = PythonOperator(
            task_id="check_drift",
            python_callable=lambda: run_cli(
                [
                    "monitor",
                    "drift",
                    "--reference-path",
                    ".rift/data/transactions.parquet",
                    "--current-path",
                    ".rift/data/transactions.parquet",
                    "--trigger-retrain",
                ]
            ),
        )
