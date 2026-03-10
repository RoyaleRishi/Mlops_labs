from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "dags" / "data"
MODEL_DIR = ROOT_DIR / "dags" / "model"
WORKING_DIR = ROOT_DIR / "working_data"

FEATURE_COLUMNS = [
    "BALANCE",
    "PURCHASES",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "TENURE",
    "CASH_ADVANCE",
]

PROFILE_NAMES = [
    "Budget Watchers",
    "Everyday Users",
    "Planned Spenders",
    "Rewards Optimizers",
    "Premium Flex",
    "High-Value Power Users",
]

PALETTE = ["#1f7a8c", "#bf4d28", "#6c8a2b", "#7a5195", "#ef5675", "#ffa600"]


def _ensure_output_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)


def _segment_description(row: pd.Series) -> str:
    if row["avg_cash_advance"] > row["avg_purchases"]:
        return "More dependent on cash advances than card purchases."
    if row["avg_credit_limit"] > 8000 and row["avg_purchases"] > 1000:
        return "High-limit customers with consistently strong monthly activity."
    if row["avg_purchases"] < 300:
        return "Lower-spend customers with conservative card usage."
    return "Balanced card users with steady spending and repayment patterns."


def _build_segment_summary(segmented_df: pd.DataFrame) -> tuple[list[dict[str, object]], dict[int, str], dict[int, str]]:
    grouped = (
        segmented_df.groupby("segment_id")
        .agg(
            customer_count=("CUST_ID", "count"),
            avg_balance=("BALANCE", "mean"),
            avg_purchases=("PURCHASES", "mean"),
            avg_credit_limit=("CREDIT_LIMIT", "mean"),
            avg_payments=("PAYMENTS", "mean"),
            avg_tenure=("TENURE", "mean"),
            avg_cash_advance=("CASH_ADVANCE", "mean"),
        )
        .reset_index()
    )

    grouped["engagement_score"] = (
        grouped["avg_purchases"].rank(method="dense")
        + grouped["avg_payments"].rank(method="dense")
        + 0.75 * grouped["avg_credit_limit"].rank(method="dense")
        - 0.50 * grouped["avg_cash_advance"].rank(method="dense")
    )
    grouped = grouped.sort_values(
        ["engagement_score", "avg_purchases", "avg_credit_limit"]
    ).reset_index(drop=True)
    grouped["profile_name"] = PROFILE_NAMES[: len(grouped)]
    grouped["color"] = PALETTE[: len(grouped)]
    grouped["description"] = grouped.apply(_segment_description, axis=1)

    label_map = {int(row.segment_id): row.profile_name for row in grouped.itertuples()}
    color_map = {int(row.segment_id): row.color for row in grouped.itertuples()}

    records: list[dict[str, object]] = []
    for row in grouped.itertuples():
        records.append(
            {
                "segment_id": int(row.segment_id),
                "profile_name": row.profile_name,
                "color": row.color,
                "description": row.description,
                "customer_count": int(row.customer_count),
                "avg_balance": round(float(row.avg_balance), 2),
                "avg_purchases": round(float(row.avg_purchases), 2),
                "avg_credit_limit": round(float(row.avg_credit_limit), 2),
                "avg_payments": round(float(row.avg_payments), 2),
                "avg_tenure": round(float(row.avg_tenure), 2),
                "avg_cash_advance": round(float(row.avg_cash_advance), 2),
            }
        )
    return records, label_map, color_map


def _build_scatter_sample(
    segmented_df: pd.DataFrame,
    projection,
    label_map: dict[int, str],
    color_map: dict[int, str],
    max_points: int = 360,
) -> list[dict[str, object]]:
    scatter_df = segmented_df[["CUST_ID", "segment_id"]].copy()
    scatter_df["pca_x"] = projection[:, 0]
    scatter_df["pca_y"] = projection[:, 1]

    segment_count = max(scatter_df["segment_id"].nunique(), 1)
    points_per_segment = max(40, max_points // segment_count)

    samples: list[dict[str, object]] = []
    for segment_id, part in scatter_df.groupby("segment_id", sort=True):
        sampled = part.sample(n=min(points_per_segment, len(part)), random_state=42)
        for row in sampled.itertuples():
            samples.append(
                {
                    "customer_id": row.CUST_ID,
                    "segment_id": int(segment_id),
                    "profile_name": label_map[int(segment_id)],
                    "color": color_map[int(segment_id)],
                    "pca_x": round(float(row.pca_x), 4),
                    "pca_y": round(float(row.pca_y), 4),
                }
            )
    return samples


def _build_scatter_svg(points: list[dict[str, object]]) -> str:
    if not points:
        return "<p>No scatter plot data available.</p>"

    width = 760
    height = 420
    padding = 40
    xs = [float(point["pca_x"]) for point in points]
    ys = [float(point["pca_y"]) for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    def scale(value: float, lower: float, upper: float, span: float) -> float:
        if upper == lower:
            return padding + span / 2
        return padding + ((value - lower) / (upper - lower)) * span

    plot_width = width - (padding * 2)
    plot_height = height - (padding * 2)

    circles = []
    for point in points:
        x = scale(float(point["pca_x"]), min_x, max_x, plot_width)
        y = height - padding - scale(float(point["pca_y"]), min_y, max_y, plot_height) + padding
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{point["color"]}" '
            f'fill-opacity="0.72"><title>{escape(point["profile_name"])} - '
            f'{escape(str(point["customer_id"]))}</title></circle>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Customer segment scatter plot">'
        f'<rect x="{padding}" y="{padding}" width="{plot_width}" height="{plot_height}" '
        'fill="#f8f5ef" stroke="#c7bda8" stroke-width="1.5" />'
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" '
        'stroke="#6d655a" stroke-width="1.2" />'
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" '
        'stroke="#6d655a" stroke-width="1.2" />'
        f'{"".join(circles)}'
        f'<text x="{width / 2:.0f}" y="{height - 8}" text-anchor="middle" fill="#4a4338" font-size="13">'
        "PCA component 1"
        "</text>"
        f'<text x="20" y="{height / 2:.0f}" text-anchor="middle" fill="#4a4338" font-size="13" '
        'transform="rotate(-90 20 {0})">'.format(height / 2)
        + "PCA component 2</text></svg>"
    )


def load_data() -> dict[str, object]:
    _ensure_output_dirs()
    source_path = DATA_DIR / "file.csv"
    dataset_path = WORKING_DIR / "customer_segmentation_source.pkl"

    source_df = pd.read_csv(source_path, usecols=["CUST_ID", *FEATURE_COLUMNS])
    source_df = source_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    source_df.to_pickle(dataset_path)

    return {
        "dataset_path": str(dataset_path),
        "record_count": int(len(source_df)),
        "source_path": str(source_path),
        "feature_columns": list(FEATURE_COLUMNS),
    }


def prepare_features(dataset_payload: dict[str, object]) -> dict[str, object]:
    _ensure_output_dirs()
    dataset_path = Path(str(dataset_payload["dataset_path"]))
    source_df = pd.read_pickle(dataset_path)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(source_df[FEATURE_COLUMNS])

    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(scaled_features)

    prepared_path = WORKING_DIR / "customer_segmentation_prepared.pkl"
    with prepared_path.open("wb") as file_handle:
        pickle.dump(
            {
                "source_df": source_df,
                "scaled_features": scaled_features,
                "projection": projection,
                "scaler": scaler,
                "feature_columns": list(FEATURE_COLUMNS),
            },
            file_handle,
        )

    return {
        "prepared_path": str(prepared_path),
        "record_count": int(len(source_df)),
        "source_path": str(dataset_payload["source_path"]),
        "feature_columns": list(FEATURE_COLUMNS),
        "explained_variance_ratio": [
            round(float(value), 4) for value in pca.explained_variance_ratio_
        ],
    }


def train_segment_model(prepared_payload: dict[str, object], filename: str) -> dict[str, object]:
    _ensure_output_dirs()
    prepared_path = Path(str(prepared_payload["prepared_path"]))
    with prepared_path.open("rb") as file_handle:
        bundle = pickle.load(file_handle)

    source_df = bundle["source_df"].copy()
    scaled_features = bundle["scaled_features"]
    projection = bundle["projection"]
    scaler = bundle["scaler"]

    candidates = []
    bic_scores = []
    for n_components in range(2, 7):
        candidate = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=3,
            random_state=42,
            reg_covar=1e-6,
        )
        candidate.fit(scaled_features)
        bic_value = float(candidate.bic(scaled_features))
        bic_scores.append({"components": int(n_components), "bic": round(bic_value, 2)})
        candidates.append((bic_value, candidate, n_components))

    _, best_model, best_components = min(candidates, key=lambda item: item[0])
    labels = best_model.predict(scaled_features)
    source_df["segment_id"] = labels

    segment_records, label_map, color_map = _build_segment_summary(source_df)
    scatter_points = _build_scatter_sample(source_df, projection, label_map, color_map)

    silhouette_value = 0.0
    if len(set(labels)) > 1:
        sample_size = min(2000, len(scaled_features))
        silhouette_value = float(
            silhouette_score(
                scaled_features,
                labels,
                sample_size=sample_size,
                random_state=42,
            )
        )

    model_path = MODEL_DIR / filename
    generated_at = datetime.now(timezone.utc).isoformat()
    with model_path.open("wb") as file_handle:
        pickle.dump(
            {
                "model": best_model,
                "scaler": scaler,
                "feature_columns": list(FEATURE_COLUMNS),
                "label_map": label_map,
                "bic_scores": bic_scores,
                "silhouette_score": round(silhouette_value, 4),
                "selected_components": int(best_components),
                "generated_at": generated_at,
            },
            file_handle,
        )

    summary_payload = {
        "source_path": str(prepared_payload["source_path"]),
        "model_path": str(model_path),
        "record_count": int(len(source_df)),
        "selected_components": int(best_components),
        "silhouette_score": round(silhouette_value, 4),
        "explained_variance_ratio": list(prepared_payload["explained_variance_ratio"]),
        "bic_scores": bic_scores,
        "segments": segment_records,
        "scatter_points": scatter_points,
        "generated_at": generated_at,
        "profile_source": str(DATA_DIR / "profile_to_score.csv"),
    }

    summary_path = WORKING_DIR / "customer_segmentation_summary.json"
    with summary_path.open("w", encoding="ascii") as file_handle:
        json.dump(summary_payload, file_handle, indent=2)

    segment_csv_path = WORKING_DIR / "customer_segmentation_segments.csv"
    pd.DataFrame(segment_records).to_csv(segment_csv_path, index=False)

    return {
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "segment_csv_path": str(segment_csv_path),
        "selected_components": int(best_components),
        "silhouette_score": round(silhouette_value, 4),
    }


def score_customer_profile(model_payload: dict[str, object]) -> dict[str, object]:
    profile_path = DATA_DIR / "profile_to_score.csv"
    profile_df = pd.read_csv(profile_path)

    with Path(str(model_payload["model_path"])).open("rb") as file_handle:
        artifact = pickle.load(file_handle)

    feature_row = profile_df.loc[0, artifact["feature_columns"]]
    scoring_frame = pd.DataFrame([feature_row.to_dict()])
    scaled_profile = artifact["scaler"].transform(scoring_frame)

    segment_id = int(artifact["model"].predict(scaled_profile)[0])
    probabilities = artifact["model"].predict_proba(scaled_profile)[0]
    confidence = float(probabilities[segment_id])

    return {
        "segment_id": segment_id,
        "profile_name": artifact["label_map"][segment_id],
        "confidence": round(confidence, 4),
        "profile_source": str(profile_path),
        "profile_input": {
            column: round(float(feature_row[column]), 2)
            for column in artifact["feature_columns"]
        },
    }


def build_dashboard(
    training_payload: dict[str, object],
    profile_result: dict[str, object],
) -> dict[str, object]:
    summary_path = Path(str(training_payload["summary_path"]))
    with summary_path.open(encoding="ascii") as file_handle:
        summary = json.load(file_handle)

    scatter_svg = _build_scatter_svg(summary["scatter_points"])

    legend_items = "".join(
        (
            '<li><span class="swatch" style="background: {color};"></span>'
            "<strong>{name}</strong><br>{description}</li>"
        ).format(
            color=escape(segment["color"]),
            name=escape(segment["profile_name"]),
            description=escape(segment["description"]),
        )
        for segment in summary["segments"]
    )

    bic_rows = "".join(
        (
            "<tr><td>{components}</td><td>{bic}</td></tr>"
        ).format(
            components=score["components"],
            bic=score["bic"],
        )
        for score in summary["bic_scores"]
    )

    segment_rows = "".join(
        (
            "<tr>"
            "<td>{name}</td>"
            "<td>{count}</td>"
            "<td>{purchases}</td>"
            "<td>{payments}</td>"
            "<td>{limit}</td>"
            "<td>{cash}</td>"
            "</tr>"
        ).format(
            name=escape(segment["profile_name"]),
            count=segment["customer_count"],
            purchases=segment["avg_purchases"],
            payments=segment["avg_payments"],
            limit=segment["avg_credit_limit"],
            cash=segment["avg_cash_advance"],
        )
        for segment in summary["segments"]
    )

    profile_rows = "".join(
        (
            "<tr><td>{feature}</td><td>{value}</td></tr>"
        ).format(feature=escape(feature), value=value)
        for feature, value in profile_result["profile_input"].items()
    )

    dashboard_path = WORKING_DIR / "customer_segmentation_dashboard.html"
    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Customer Segmentation Dashboard</title>
  <style>
    :root {{
      --paper: #f4efe4;
      --ink: #2a261f;
      --accent: #1f7a8c;
      --card: #fffaf1;
      --line: #d7ccb5;
      --muted: #645d52;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #e6d7b5 0, transparent 35%),
        linear-gradient(180deg, #efe6d1 0%, #f7f1e6 55%, #efe8dc 100%);
      min-height: 100vh;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto 40px;
    }}
    .hero {{
      background: rgba(255, 250, 241, 0.88);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 24px;
      box-shadow: 0 18px 48px rgba(50, 38, 12, 0.08);
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ margin: 0; line-height: 1.55; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(50, 38, 12, 0.05);
    }}
    .metric {{
      font-size: 2rem;
      display: block;
      margin-top: 8px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-top: 18px;
    }}
    ul {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 12px;
    }}
    li {{
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    .swatch {{
      display: inline-block;
      width: 14px;
      height: 14px;
      border-radius: 999px;
      margin-right: 8px;
      vertical-align: middle;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.82rem;
      background: #ebe2cf;
      color: #4f473c;
    }}
    .footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    @media (max-width: 860px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <span class="pill">Custom lab variant</span>
      <h1>Customer Segmentation Dashboard</h1>
      <p>
        This lab variant replaces the original KMeans elbow workflow with a Gaussian Mixture
        segmentation model selected by BIC, then publishes an HTML dashboard with segment
        summaries and a scored customer profile.
      </p>
      <div class="grid">
        <article class="card">
          <strong>Customers analyzed</strong>
          <span class="metric">{summary["record_count"]}</span>
        </article>
        <article class="card">
          <strong>Selected segments</strong>
          <span class="metric">{summary["selected_components"]}</span>
        </article>
        <article class="card">
          <strong>Silhouette score</strong>
          <span class="metric">{summary["silhouette_score"]}</span>
        </article>
        <article class="card">
          <strong>Predicted profile</strong>
          <span class="metric">{escape(profile_result["profile_name"])}</span>
        </article>
      </div>
    </section>

    <section class="layout">
      <article class="card">
        <h2>Segment Map</h2>
        <p>The plot below shows sampled customers projected into two PCA dimensions for quick visual inspection.</p>
        <div style="margin-top: 16px;">{scatter_svg}</div>
      </article>
      <article class="card">
        <h2>Segment Legend</h2>
        <ul>{legend_items}</ul>
      </article>
    </section>

    <section class="layout">
      <article class="card">
        <h2>Model Selection</h2>
        <p>BIC was used to choose the number of Gaussian mixture components.</p>
        <table>
          <thead>
            <tr><th>Components</th><th>BIC</th></tr>
          </thead>
          <tbody>{bic_rows}</tbody>
        </table>
      </article>
      <article class="card">
        <h2>Scored Profile</h2>
        <p>
          The custom profile in <code>{escape(profile_result["profile_source"])}</code> was assigned to
          <strong>{escape(profile_result["profile_name"])}</strong> with confidence
          <strong>{profile_result["confidence"]}</strong>.
        </p>
        <table style="margin-top: 12px;">
          <thead>
            <tr><th>Feature</th><th>Value</th></tr>
          </thead>
          <tbody>{profile_rows}</tbody>
        </table>
      </article>
    </section>

    <section class="card" style="margin-top: 18px;">
      <h2>Segment Summary Table</h2>
      <table>
        <thead>
          <tr>
            <th>Profile</th>
            <th>Customers</th>
            <th>Avg purchases</th>
            <th>Avg payments</th>
            <th>Avg credit limit</th>
            <th>Avg cash advance</th>
          </tr>
        </thead>
        <tbody>{segment_rows}</tbody>
      </table>
      <p class="footer">
        Generated at {escape(summary["generated_at"])} from <code>{escape(summary["source_path"])}</code>.
        The dashboard artifact is written to <code>{escape(str(dashboard_path))}</code>.
      </p>
    </section>
  </main>
</body>
</html>
"""
    dashboard_path.write_text(dashboard_html, encoding="ascii")

    return {
        "dashboard_path": str(dashboard_path),
        "summary_path": str(summary_path),
        "predicted_profile": profile_result["profile_name"],
    }
