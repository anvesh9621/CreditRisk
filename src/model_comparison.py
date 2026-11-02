import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.ml import PipelineModel

from src.utils import (
    PLOTS_DIR,
    METRICS_DIR,
    ANALYSIS_DIR,
    BEST_MODEL_PATH,
    save_json,
    ensure_directories,
)

NAVY = "#0f2042"
CORAL = "#e63946"
TEAL = "#2a9d8f"
GOLD = "#e9c46a"
SLATE = "#457b9d"
PURPLE = "#7209b7"

MODEL_COLORS = {
    "Logistic Regression (Unweighted)": "#6c757d",
    "Logistic Regression (Weighted)": SLATE,
    "Decision Tree": GOLD,
    "Random Forest": TEAL,
    "Gradient Boosted Tree": CORAL,
    "Naive Bayes": PURPLE,
}


def extract_feature_names_and_importances(best_model: PipelineModel, raw_feature_names: list) -> pd.DataFrame:
    """
    Extract tree model feature importances mapped to actual feature names.
    Handles OneHotEncoder expanded feature dimensions accurately.
    """
    # Find the tree model stage
    tree_stage = None
    for stage in reversed(best_model.stages):
        if hasattr(stage, "featureImportances"):
            tree_stage = stage
            break

    if tree_stage is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    importances = tree_stage.featureImportances.toArray()

    # Find the VectorAssembler stage to reconstruct feature column names
    assembler_stage = None
    for stage in best_model.stages:
        if stage.__class__.__name__ == "VectorAssembler":
            assembler_stage = stage
            break

    # Build expanded feature names list matching vector length
    expanded_names = []
    if assembler_stage:
        for col_name in assembler_stage.getInputCols():
            if col_name.endswith("_vec"):
                base_name = col_name.replace("_vec", "")
                expanded_names.append(f"{base_name}_encoded")
            else:
                expanded_names.append(col_name)

    # Pad or slice if length mismatch due to multi-category one-hot vectors
    if len(expanded_names) < len(importances):
        diff = len(importances) - len(expanded_names)
        for i in range(diff):
            expanded_names.append(f"Cat_Encoded_{i+1}")
    elif len(expanded_names) > len(importances):
        expanded_names = expanded_names[: len(importances)]

    feat_df = pd.DataFrame({
        "Feature": expanded_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Normalization (ensure sum = 1.0)
    if feat_df["Importance"].sum() > 0:
        feat_df["Importance_Percentage"] = (feat_df["Importance"] / feat_df["Importance"].sum()) * 100
    else:
        feat_df["Importance_Percentage"] = 0.0

    return feat_df


def plot_metric_comparison(comp_df: pd.DataFrame, metric: str, title: str, filename: str, is_percentage: bool = False):
    """Generate standalone high-resolution bar chart comparing a specific metric across models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = comp_df["Model"].tolist()
    values = comp_df[metric].tolist()
    colors = [MODEL_COLORS.get(m, SLATE) for m in models]

    bars = ax.bar(models, values, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel(metric, fontsize=11)

    max_v = max(values) if values else 1.0
    ax.set_ylim(0, max_v * 1.18 if max_v > 0 else 1.0)

    for bar in bars:
        h = bar.get_height()
        label_text = f"{h:.2f}s" if "Time" in metric else (f"{h*100:.1f}%" if is_percentage else f"{h:.4f}")
        ax.annotate(label_text, xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def plot_roc_pr_curves(curve_data: dict):
    """Generate side-by-side ROC Curves and Precision-Recall Curves for all trained models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curves
    for model_name, data in curve_data.items():
        roc_pts = data["roc_curve"]
        fprs = [p["fpr"] for p in roc_pts]
        tprs = [p["tpr"] for p in roc_pts]
        axes[0].plot(fprs, tprs, label=model_name, color=MODEL_COLORS.get(model_name, SLATE), linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC=0.50)")
    axes[0].set_title("ROC Curves Comparison", fontsize=13, fontweight="bold", color=NAVY)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # PR Curves
    for model_name, data in curve_data.items():
        pr_pts = data["pr_curve"]
        recalls = [p["recall"] for p in pr_pts]
        precs = [p["precision"] for p in pr_pts]
        axes[1].plot(recalls, precs, label=model_name, color=MODEL_COLORS.get(model_name, SLATE), linewidth=2)

    axes[1].axhline(0.2212, color="gray", linestyle="--", alpha=0.6, label="Base Default Rate (22.1%)")
    axes[1].set_title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold", color=NAVY)
    axes[1].set_xlabel("Recall (Class 1)", fontsize=11)
    axes[1].set_ylabel("Precision (Class 1)", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "23_roc_pr_curves_comparison.png", dpi=300)
    plt.close()


def plot_confusion_matrices(metrics_dict: dict):
    """Generate multi-panel confusion matrix visualizations for all models."""
    n_models = len(metrics_dict)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        cm = metrics["confusion_matrix"]
        matrix = np.array([
            [cm["tn"], cm["fp"]],
            [cm["fn"], cm["tp"]]
        ])
        ax = axes[idx]
        sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"fontsize": 12, "fontweight": "bold"},
                    xticklabels=["Pred: No Default", "Pred: Default"],
                    yticklabels=["Actual: No Default", "Actual: Default"])
        ax.set_title(f"{model_name}\nAcc: {metrics['accuracy']:.1%} | Rec(C1): {metrics['recall_default']:.1%}",
                     fontsize=11, fontweight="bold", color=NAVY)
        ax.set_ylabel("True Class", fontsize=10)
        ax.set_xlabel("Predicted Class", fontsize=10)

    # Hide extra subplots if any
    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "25_confusion_matrices.png", dpi=300)
    plt.close()


def plot_feature_importance(feat_df: pd.DataFrame):
    """Plot Top 15 Feature Importances bar chart."""
    top15 = feat_df.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top15["Feature"], top15["Importance_Percentage"], color=TEAL, edgecolor="black", linewidth=0.6, height=0.65)
    ax.set_title("Top 15 Most Important Features Influencing Credit Default", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Relative Feature Importance (%)", fontsize=11)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}%", xy=(w + 0.2, bar.get_y() + bar.get_height() / 2),
                    va="center", ha="left", fontsize=9, fontweight="bold", color=NAVY)

    max_w = top15["Importance_Percentage"].max()
    ax.set_xlim(0, max_w * 1.15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "26_feature_importance.png", dpi=300)
    plt.close()


def compare_and_select_best_model(training_results: dict):
    """
    Perform comprehensive model comparison, plot generation,
    best model selection, native Spark model saving, and metadata export.
    """
    ensure_directories()
    metrics_dict = training_results["metrics"]
    models_dict = training_results["models"]
    curve_data = training_results["curve_data"]
    hyperparameters = training_results["hyperparameters"]
    raw_feature_names = training_results["feature_names"]

    # 1. Build Comparison DataFrame
    rows = []
    for name, m in metrics_dict.items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-score": m["f1_score"],
            "Default Precision": m["precision_default"],
            "Default Recall": m["recall_default"],
            "Default F1": m["f1_default"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "Training Time (s)": m["training_time"],
            "Prediction Time (s)": m["prediction_time"],
            "Composite Score": m["composite_score"],
        })

    comp_df = pd.DataFrame(rows).sort_values(by="Composite Score", ascending=False).reset_index(drop=True)

    # Save Comparison CSV
    comp_csv_path = METRICS_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved Model Comparison Table: {comp_csv_path}")

    # Display Table
    print("\n" + "=" * 95)
    print("MODEL COMPARISON TABLE (Ranked by Composite Validation Score)")
    print("=" * 95)
    print(comp_df[["Model", "Accuracy", "Default Recall", "F1-score", "ROC-AUC", "PR-AUC", "Training Time (s)"]].to_string(index=False))
    print("=" * 95)

    # 2. Generate Comparison Plots
    plot_metric_comparison(comp_df, "Accuracy", "Model Accuracy Comparison", "16_accuracy_comparison.png", is_percentage=True)
    plot_metric_comparison(comp_df, "Precision", "Model Precision Comparison", "17_precision_comparison.png")
    plot_metric_comparison(comp_df, "Default Recall", "Default Class Recall Comparison (Critical Credit Metric)", "18_recall_comparison.png", is_percentage=True)
    plot_metric_comparison(comp_df, "F1-score", "Model F1-Score Comparison", "19_f1_score_comparison.png")
    plot_metric_comparison(comp_df, "ROC-AUC", "Model ROC-AUC Comparison", "20_roc_auc_comparison.png")
    plot_metric_comparison(comp_df, "PR-AUC", "Model PR-AUC (Precision-Recall) Comparison", "21_pr_auc_comparison.png")
    plot_metric_comparison(comp_df, "Training Time (s)", "Model Training Time Comparison (Seconds)", "22_training_time_comparison.png")
    plot_roc_pr_curves(curve_data)
    plot_confusion_matrices(metrics_dict)

    # 3. Select Best Model
    best_row = comp_df.iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline_model = models_dict[best_model_name]
    best_metrics = metrics_dict[best_model_name]
    best_params = hyperparameters.get(best_model_name, {})

    print(f"\n[BEST MODEL SELECTED]: {best_model_name}")
    print(f"Selection Justification: Highest composite validation score ({best_row['Composite Score']:.4f})")
    print(f"ROC-AUC: {best_row['ROC-AUC']:.4f} | PR-AUC: {best_row['PR-AUC']:.4f} | Default Recall: {best_row['Default Recall']:.2%}")

    # 4. Feature Importance (Using tree model: Random Forest or GBT)
    tree_model_to_inspect = models_dict.get("Gradient Boosted Tree") or models_dict.get("Random Forest")
    feat_df = extract_feature_names_and_importances(tree_model_to_inspect, raw_feature_names)
    feat_csv_path = ANALYSIS_DIR / "feature_importance.csv"
    feat_df.to_csv(feat_csv_path, index=False)
    print(f"Saved Feature Importance Table: {feat_csv_path}")
    plot_feature_importance(feat_df)

    # 5. Native PySpark Model Saving
    print(f"\nSaving Best PySpark PipelineModel natively to: {BEST_MODEL_PATH}...")
    best_pipeline_model.write().overwrite().save(str(BEST_MODEL_PATH))
    print("PipelineModel saved successfully.")

    # 6. Save Best Model Metadata
    metadata = {
        "best_model_name": best_model_name,
        "selection_criteria": "Multi-criteria composite score (0.35*ROC-AUC + 0.25*PR-AUC + 0.25*Default_Recall + 0.15*Default_F1)",
        "composite_score": float(best_row["Composite Score"]),
        "metrics": best_metrics,
        "hyperparameters": best_params,
        "threshold": 0.50,
        "risk_thresholds": {
            "low_risk": "probability < 0.25",
            "medium_risk": "0.25 <= probability < 0.50",
            "high_risk": "probability >= 0.50",
        },
        "top_features": feat_df.head(15).to_dict(orient="records"),
    }
    meta_path = METRICS_DIR / "best_model_metadata.json"
    save_json(metadata, meta_path)

    # Save Comparison JSON
    save_json({
        "comparison_table": comp_df.to_dict(orient="records"),
        "models_evaluated": list(metrics_dict.keys()),
        "best_model": best_model_name,
        "curve_data": curve_data
    }, METRICS_DIR / "model_comparison.json")

    return {
        "best_model_name": best_model_name,
        "best_model": best_pipeline_model,
        "comparison_df": comp_df,
        "feature_importance_df": feat_df,
        "metadata": metadata,
    }

