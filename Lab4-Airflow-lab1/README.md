# Lab Changes

- Replaced the original `KMeans + elbow` workflow with a `GaussianMixture` segmentation pipeline.
- Added model selection using `BIC` across multiple segment counts.
- Added a custom scoring input file at `dags/data/profile_to_score.csv`.
- Added artifact outputs in `working_data/` instead of only passing data through XCom.
- Added an HTML dashboard output at `working_data/customer_segmentation_dashboard.html`.
- Renamed the DAG to `customer_segmentation_dashboard_lab`.

# How to Run

1. Start Docker Desktop.
2. Initialize Airflow:

```bash
docker compose up airflow-init
```

3. Start Airflow:

```bash
docker compose up
```

4. Open `http://localhost:8080`.
5. Log in with:
   Username: `airflow2`
   Password: `airflow2`
6. Trigger the DAG `customer_segmentation_dashboard_lab`.

# Outputs

After the DAG finishes, check:

- `working_data/customer_segmentation_dashboard.html`
- `working_data/customer_segmentation_summary.json`
- `working_data/customer_segmentation_segments.csv`
- `dags/model/customer_segmentation_model.pkl`
