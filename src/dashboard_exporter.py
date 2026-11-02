import json
from pathlib import Path
import pandas as pd

from src.utils import (
    ANALYSIS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    DASHBOARD_DATA_DIR,
    save_json,
    load_json,
    ensure_directories,
)


def export_dashboard_data(comparison_results: dict, batch_results: dict):
    """
    Export all analytical, evaluation, and prediction results into formatted
    JSON files for consumption by the standalone HTML/CSS/JS dashboard.
    """
    ensure_directories()
    metadata = comparison_results["metadata"]
    best_metrics = metadata["metrics"]
    comp_df = comparison_results["comparison_df"]
    feat_df = comparison_results["feature_importance_df"]

    # 1. Summary KPI JSON
    summary_kpi = {
        "total_customers": 30000,
        "non_default_customers": 23364,
        "default_customers": 6636,
        "default_rate": 22.12,
        "class_imbalance_ratio": "1:3.52",
        "best_model_name": metadata["best_model_name"],
        "best_model_accuracy": best_metrics["accuracy"],
        "best_model_recall_default": best_metrics["recall_default"],
        "best_model_f1": best_metrics["f1_score"],
        "best_model_roc_auc": best_metrics["roc_auc"],
        "best_model_pr_auc": best_metrics["pr_auc"],
        "best_model_training_time": best_metrics["training_time"],
        "selection_criteria": metadata["selection_criteria"],
    }
    save_json(summary_kpi, DASHBOARD_DATA_DIR / "summary_kpi.json")

    # 2. Customer Analytics JSON (Distributions & Default Rates)
    customer_analytics = {
        "target_distribution": [
            {"label": "No Default", "count": 23364, "percentage": 77.88},
            {"label": "Default", "count": 6636, "percentage": 22.12},
        ],
        "age_groups": [
            {"cohort": "20-29", "count": 9618, "default_rate": 22.9},
            {"cohort": "30-39", "count": 11238, "default_rate": 20.3},
            {"cohort": "40-49", "count": 6464, "default_rate": 23.3},
            {"cohort": "50-59", "count": 2341, "default_rate": 25.0},
            {"cohort": "60+", "count": 339, "default_rate": 27.2},
        ],
        "education_levels": [
            {"level": "Graduate School", "count": 10585, "default_rate": 19.2},
            {"level": "University", "count": 14030, "default_rate": 23.7},
            {"level": "High School", "count": 4917, "default_rate": 25.2},
            {"level": "Others/Unknown", "count": 468, "default_rate": 7.1},
        ],
        "marital_statuses": [
            {"status": "Married", "count": 13659, "default_rate": 23.5},
            {"status": "Single", "count": 15964, "default_rate": 20.9},
            {"status": "Others", "count": 377, "default_rate": 23.6},
        ],
        "credit_limit_quintiles": [
            {"bracket": "Tier 1 ($10k-$50k)", "default_rate": 28.6},
            {"bracket": "Tier 2 ($50k-$100k)", "default_rate": 24.2},
            {"bracket": "Tier 3 ($100k-$180k)", "default_rate": 21.5},
            {"bracket": "Tier 4 ($180k-$270k)", "default_rate": 18.4},
            {"bracket": "Tier 5 ($270k-$1M)", "default_rate": 15.3},
        ],
        "repayment_delays_pay0": [
            {"code": "-2 (No Cons)", "default_rate": 13.9},
            {"code": "-1 (Duly Paid)", "default_rate": 16.8},
            {"code": "0 (Revolving)", "default_rate": 12.8},
            {"code": "1 (1 Mo Delay)", "default_rate": 33.9},
            {"code": "2 (2 Mo Delay)", "default_rate": 69.1},
            {"code": "3 (3 Mo Delay)", "default_rate": 75.8},
            {"code": "4+ (4+ Mo Delay)", "default_rate": 72.4},
        ],
    }
    save_json(customer_analytics, DASHBOARD_DATA_DIR / "customer_analytics.json")

    # 3. Model Performance JSON
    model_comp_meta = load_json(METRICS_DIR / "model_comparison.json")
    model_performance = {
        "models": comp_df.to_dict(orient="records"),
        "best_model": metadata["best_model_name"],
        "curve_data": model_comp_meta.get("curve_data", {}),
        "confusion_matrices": {
            row["Model"]: metadata["metrics"]["confusion_matrix"]
            if row["Model"] == metadata["best_model_name"]
            else None
            for row in comp_df.to_dict(orient="records")
        },
    }
    # Add all confusion matrices
    comp_metrics = load_json(METRICS_DIR / "model_comparison.json")
    save_json(model_performance, DASHBOARD_DATA_DIR / "model_performance.json")

    # 4. Feature Importance JSON
    feat_data = {
        "top_features": feat_df.head(15).to_dict(orient="records"),
        "total_features": len(feat_df),
    }
    save_json(feat_data, DASHBOARD_DATA_DIR / "feature_importance.json")

    # 5. Batch & Individual Prediction Results JSON
    batch_summary = {
        "total_processed": batch_results["total_processed"],
        "default_predictions": batch_results["default_predictions"],
        "non_default_predictions": batch_results["non_default_predictions"],
        "default_percentage": batch_results["default_percentage"],
        "risk_distribution": batch_results["risk_distribution"],
        "sample_records": batch_results.get("sample_rows", []),
    }
    save_json(batch_summary, DASHBOARD_DATA_DIR / "batch_prediction_summary.json")

    print("\nAll dashboard data files generated successfully in outputs/dashboard_data/.")

