"""Streamlit dashboard for Rift monitoring (drift, quality, experiment tracking)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import streamlit as st

    st.set_page_config(page_title="Rift Monitoring", page_icon="🔍", layout="wide")
    st.title("Rift Fraud Detection - Monitoring Dashboard")

    tab1, tab2, tab3 = st.tabs(["Data Drift", "Model Quality", "Experiment History"])

    with tab1:
        st.header("Data Drift Analysis")
        st.write("Compare reference (training) data against current (production) data.")

        ref_path = st.text_input("Reference dataset path", "data/reference.parquet", key="drift_ref")
        cur_path = st.text_input("Current dataset path", "data/current.parquet", key="drift_cur")

        if st.button("Generate Drift Report", key="drift_btn"):
            from monitoring.evidently_dashboard import generate_drift_report

            with st.spinner("Generating Evidently drift report..."):
                result = generate_drift_report(ref_path, cur_path)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(f"Report saved to: {result['report_path']}")
                if result.get("dataset_drift_detected"):
                    st.warning(f"Drift detected in {result['n_drifted_features']} features")
                else:
                    st.info("No significant drift detected")

                report_path = Path(result["report_path"])
                if report_path.exists():
                    st.components.v1.html(report_path.read_text(), height=800, scrolling=True)

    with tab2:
        st.header("Data Validation (Deepchecks)")

        ref_path_v = st.text_input("Reference dataset", "data/reference.parquet", key="val_ref")
        cur_path_v = st.text_input("Current dataset", "data/current.parquet", key="val_cur")

        if st.button("Run Validation Suite", key="val_btn"):
            from validate.deepchecks_suite import run_data_validation

            with st.spinner("Running Deepchecks validation..."):
                result = run_data_validation(ref_path_v, cur_path_v)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    status = "Passed" if result["integrity_passed"] else "Failed"
                    st.metric("Data Integrity", status)
                with col2:
                    status = "Passed" if result["validation_passed"] else "Failed"
                    st.metric("Train/Test Validation", status)

                report_path = Path(result["report_path"])
                if report_path.exists():
                    st.components.v1.html(report_path.read_text(), height=800, scrolling=True)

    with tab3:
        st.header("Experiment History (MLflow)")
        st.write("View and compare training runs.")

        if st.button("Load Experiments", key="exp_btn"):
            from monitoring.mlflow_setup import get_experiment_summary

            runs = get_experiment_summary()
            if runs:
                import pandas as pd

                df = pd.DataFrame([
                    {
                        "run_id": r["run_id"][:8],
                        "status": r["status"],
                        **{k: f"{v:.4f}" for k, v in r["metrics"].items() if isinstance(v, float)},
                        **r["params"],
                    }
                    for r in runs
                ])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No experiment runs found. Train a model first.")

except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    print("Then: streamlit run src/monitoring/streamlit_app.py")
