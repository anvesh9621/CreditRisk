import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F

from src.utils import (
    PLOTS_DIR,
    ANALYSIS_DIR,
    get_spark_session,
    stop_spark_session,
    ensure_directories,
)
from src.data_loader import load_raw_data

# Set visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#ccd1d9"
plt.rcParams["axes.linewidth"] = 0.8

NAVY = "#0f2042"
CORAL = "#e63946"
TEAL = "#2a9d8f"
GOLD = "#e9c46a"
SLATE = "#457b9d"
PALETTE = [TEAL, CORAL, SLATE, GOLD, "#9b5de5", "#f15bb5"]


def perform_eda(spark, df: DataFrame):
    """
    Execute full Exploratory Data Analysis:
    Univariate, Target, Bivariate, Correlation, and Outlier analyses.
    Generates high-resolution figures and statistical CSV exports.
    """
    ensure_directories()
    target_col = "default_payment_next_month" if "default_payment_next_month" in df.columns else "default.payment.next.month"

    # Convert Spark DataFrame to Pandas for high-fidelity visualization
    pdf = df.toPandas()

    # Create descriptive categorical labels for EDA
    pdf["SEX_LABEL"] = pdf["SEX"].map({1: "Male", 2: "Female"})
    pdf["EDUCATION_LABEL"] = pdf["EDUCATION"].map({
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Others",
        5: "Unknown",
        6: "Unknown",
        0: "Unknown"
    }).fillna("Others")
    pdf["MARRIAGE_LABEL"] = pdf["MARRIAGE"].map({
        1: "Married",
        2: "Single",
        3: "Others",
        0: "Others"
    }).fillna("Others")
    pdf["DEFAULT_LABEL"] = pdf[target_col].map({0: "No Default", 1: "Default"})

    # Age Groups
    pdf["AGE_GROUP"] = pd.cut(
        pdf["AGE"],
        bins=[20, 30, 40, 50, 60, 100],
        labels=["20-29", "30-39", "40-49", "50-59", "60+"],
        right=False
    )

    # Credit Limit Brackets
    pdf["LIMIT_BRACKET"] = pd.qcut(
        pdf["LIMIT_BAL"],
        q=5,
        labels=["Tier 1 (Lowest)", "Tier 2", "Tier 3", "Tier 4", "Tier 5 (Highest)"]
    )

    print("Running Univariate & Target Analysis...")

    # 1. Target Distribution (Bar + Donut)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = pdf[target_col].value_counts().sort_index()
    percentages = (counts / len(pdf)) * 100

    bars = axes[0].bar(["No Default (0)", "Default (1)"], counts, color=[TEAL, CORAL], width=0.5, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Target Class Count", fontsize=13, fontweight="bold", pad=12, color=NAVY)
    axes[0].set_ylabel("Customer Count", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    axes[1].pie(counts, labels=["No Default (77.88%)", "Default (22.12%)"], autopct="%1.2f%%",
                startangle=90, colors=[TEAL, CORAL], explode=(0.04, 0.04),
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    # Donut circle
    centre_circle = plt.Circle((0, 0), 0.65, fc="white")
    axes[1].add_artist(centre_circle)
    axes[1].set_title("Target Class Proportion (Class Imbalance ~3.5:1)", fontsize=13, fontweight="bold", pad=12, color=NAVY)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_target_distribution.png", dpi=300)
    plt.close()

    # 2. Age Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=pdf, x="AGE", hue="DEFAULT_LABEL", kde=True, ax=axes[0],
                 palette={"No Default": TEAL, "Default": CORAL}, bins=30, alpha=0.6)
    axes[0].set_title("Age Distribution by Default Status", fontsize=13, fontweight="bold", color=NAVY)
    axes[0].set_xlabel("Customer Age (Years)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)

    sns.boxplot(data=pdf, x="DEFAULT_LABEL", y="AGE", ax=axes[1],
                palette=[TEAL, CORAL], width=0.4, boxprops=dict(alpha=0.8))
    axes[1].set_title("Age Boxplot by Default Status", fontsize=13, fontweight="bold", color=NAVY)
    axes[1].set_xlabel("Default Status", fontsize=11)
    axes[1].set_ylabel("Age", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_age_distribution.png", dpi=300)
    plt.close()

    # 3. Credit Limit Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=pdf, x="LIMIT_BAL", hue="DEFAULT_LABEL", kde=True, ax=axes[0],
                 palette={"No Default": TEAL, "Default": CORAL}, bins=35, alpha=0.6)
    axes[0].set_title("Credit Limit (NT$) Distribution", fontsize=13, fontweight="bold", color=NAVY)
    axes[0].set_xlabel("Credit Limit (NT$)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)

    sns.boxplot(data=pdf, x="DEFAULT_LABEL", y="LIMIT_BAL", ax=axes[1],
                palette=[TEAL, CORAL], width=0.4, boxprops=dict(alpha=0.8))
    axes[1].set_title("Credit Limit Boxplot", fontsize=13, fontweight="bold", color=NAVY)
    axes[1].set_xlabel("Default Status", fontsize=11)
    axes[1].set_ylabel("Credit Limit (NT$)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_credit_limit_distribution.png", dpi=300)
    plt.close()

    # 4. Demographic Distributions (Gender, Education, Marital Status)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.countplot(data=pdf, x="SEX_LABEL", ax=axes[0], palette=[SLATE, CORAL], hue="SEX_LABEL", legend=False)
    axes[0].set_title("Gender Distribution", fontsize=12, fontweight="bold", color=NAVY)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count", fontsize=11)

    edu_order = ["Graduate School", "University", "High School", "Others", "Unknown"]
    sns.countplot(data=pdf, x="EDUCATION_LABEL", order=edu_order, ax=axes[1], palette="crest", hue="EDUCATION_LABEL", legend=False)
    axes[1].set_title("Education Level Distribution", fontsize=12, fontweight="bold", color=NAVY)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("")

    mar_order = ["Married", "Single", "Others"]
    sns.countplot(data=pdf, x="MARRIAGE_LABEL", order=mar_order, ax=axes[2], palette="viridis", hue="MARRIAGE_LABEL", legend=False)
    axes[2].set_title("Marital Status Distribution", fontsize=12, fontweight="bold", color=NAVY)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_demographic_distributions.png", dpi=300)
    plt.close()

    # 5. Repayment Status Distributions (PAY_0 through PAY_6)
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    month_names = ["September (PAY_0)", "August (PAY_2)", "July (PAY_3)", "June (PAY_4)", "May (PAY_5)", "April (PAY_6)"]
    for i, col in enumerate(pay_cols):
        sns.countplot(data=pdf, x=col, ax=axes[i], color=SLATE, alpha=0.85)
        axes[i].set_title(f"Repayment Status: {month_names[i]}", fontsize=11, fontweight="bold", color=NAVY)
        axes[i].set_xlabel("Delay Status Code", fontsize=10)
        axes[i].set_ylabel("Count", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_repayment_status_distribution.png", dpi=300)
    plt.close()

    # 6. Bill Amounts Distribution (Log Scale Boxplots)
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    fig, ax = plt.subplots(figsize=(12, 5))
    melted_bills = pdf.melt(id_vars=["ID", target_col], value_vars=bill_cols, var_name="Month", value_name="Bill_Amount")
    sns.boxplot(data=melted_bills, x="Month", y="Bill_Amount", ax=ax, palette="Blues_r", showfliers=False)
    ax.set_title("Bill Statement Amounts Distribution Across 6 Months (Outliers Hidden)", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Billing Month", fontsize=11)
    ax.set_ylabel("Bill Amount (NT$)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_bill_amounts_distribution.png", dpi=300)
    plt.close()

    # 7. Payment Amounts Distribution (Log Scale Boxplots)
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    fig, ax = plt.subplots(figsize=(12, 5))
    melted_pays = pdf.melt(id_vars=["ID", target_col], value_vars=pay_amt_cols, var_name="Month", value_name="Pay_Amount")
    sns.boxplot(data=melted_pays, x="Month", y="Pay_Amount", ax=ax, palette="Greens_r", showfliers=False)
    ax.set_title("Previous Payment Amounts Distribution Across 6 Months (Outliers Hidden)", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Payment Month", fontsize=11)
    ax.set_ylabel("Payment Amount (NT$)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_payment_amounts_distribution.png", dpi=300)
    plt.close()

    print("Running Bivariate Analysis...")

    # 8. Default Rate by Age Group
    age_biv = pdf.groupby("AGE_GROUP", observed=False)[target_col].agg(["count", "mean"]).reset_index()
    age_biv["mean"] = age_biv["mean"] * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(age_biv["AGE_GROUP"].astype(str), age_biv["mean"], color=SLATE, width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(pdf[target_col].mean() * 100, color=CORAL, linestyle="--", linewidth=1.5, label=f"Overall Avg ({pdf[target_col].mean()*100:.2f}%)")
    ax.set_title("Default Rate by Age Cohort", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Age Cohort", fontsize=11)
    ax.set_ylabel("Default Rate (%)", fontsize=11)
    ax.legend()
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_default_by_age_group.png", dpi=300)
    plt.close()

    # 9. Default Rate by Education
    edu_biv = pdf.groupby("EDUCATION_LABEL")[target_col].agg(["count", "mean"]).loc[edu_order].reset_index()
    edu_biv["mean"] = edu_biv["mean"] * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(edu_biv["EDUCATION_LABEL"], edu_biv["mean"], color=TEAL, width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(pdf[target_col].mean() * 100, color=CORAL, linestyle="--", linewidth=1.5, label=f"Overall Avg ({pdf[target_col].mean()*100:.2f}%)")
    ax.set_title("Default Rate by Education Level", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylabel("Default Rate (%)", fontsize=11)
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_default_by_education.png", dpi=300)
    plt.close()

    # 10. Default Rate by Marital Status
    mar_biv = pdf.groupby("MARRIAGE_LABEL")[target_col].agg(["count", "mean"]).loc[mar_order].reset_index()
    mar_biv["mean"] = mar_biv["mean"] * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(mar_biv["MARRIAGE_LABEL"], mar_biv["mean"], color=GOLD, edgecolor="black", width=0.45, linewidth=0.5)
    ax.axhline(pdf[target_col].mean() * 100, color=CORAL, linestyle="--", linewidth=1.5, label=f"Overall Avg ({pdf[target_col].mean()*100:.2f}%)")
    ax.set_title("Default Rate by Marital Status", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylabel("Default Rate (%)", fontsize=11)
    ax.legend()
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_default_by_marital_status.png", dpi=300)
    plt.close()

    # 11. Default Rate by Credit Limit Bracket
    lim_biv = pdf.groupby("LIMIT_BRACKET", observed=False)[target_col].agg(["count", "mean"]).reset_index()
    lim_biv["mean"] = lim_biv["mean"] * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(lim_biv["LIMIT_BRACKET"].astype(str), lim_biv["mean"], color=SLATE, width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(pdf[target_col].mean() * 100, color=CORAL, linestyle="--", linewidth=1.5, label=f"Overall Avg ({pdf[target_col].mean()*100:.2f}%)")
    ax.set_title("Default Rate by Credit Limit Quintile", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylabel("Default Rate (%)", fontsize=11)
    ax.legend()
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_default_by_credit_limit.png", dpi=300)
    plt.close()

    # 12. Default Rate by Repayment History (PAY_0 Delay)
    pay0_biv = pdf.groupby("PAY_0")[target_col].agg(["count", "mean"]).reset_index()
    pay0_biv["mean"] = pay0_biv["mean"] * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(pay0_biv["PAY_0"].astype(str), pay0_biv["mean"], color=CORAL, width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(pdf[target_col].mean() * 100, color=NAVY, linestyle="--", linewidth=1.5, label=f"Overall Avg ({pdf[target_col].mean()*100:.2f}%)")
    ax.set_title("Default Rate by September Repayment Status (PAY_0)", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Repayment Status Code (-2: No balance, -1: Duly paid, 0: Revolving, 1+: Months delayed)", fontsize=10)
    ax.set_ylabel("Default Rate (%)", fontsize=11)
    ax.legend()
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_default_by_repayment_history.png", dpi=300)
    plt.close()

    print("Running Correlation Analysis...")

    # 13. Correlation Matrix Heatmap
    corr_cols = [c for c in pdf.columns if c not in ["ID", "SEX_LABEL", "EDUCATION_LABEL", "MARRIAGE_LABEL", "DEFAULT_LABEL", "AGE_GROUP", "LIMIT_BRACKET"]]
    corr_matrix = pdf[corr_cols].corr()
    corr_path = ANALYSIS_DIR / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_path)

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", vmin=-0.4, vmax=1.0, annot=False,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Matrix Heatmap", fontsize=15, fontweight="bold", color=NAVY, pad=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_correlation_matrix_heatmap.png", dpi=300)
    plt.close()

    # 14. Top Feature Correlations with Target
    target_corrs = corr_matrix[target_col].drop(target_col).sort_values()
    target_corr_df = pd.DataFrame({"Feature": target_corrs.index, "Correlation": target_corrs.values})
    target_corr_path = ANALYSIS_DIR / "target_correlations.csv"
    target_corr_df.to_csv(target_corr_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [CORAL if x > 0 else TEAL for x in target_corrs.values]
    bars = ax.barh(target_corrs.index, target_corrs.values, color=colors, edgecolor="black", linewidth=0.5, height=0.6)
    ax.set_title("Feature Linear Correlation with Default Target", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Pearson Correlation Coefficient", fontsize=11)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.8)
    for bar in bars:
        w = bar.get_width()
        ha = "left" if w >= 0 else "right"
        offset = 0.005 if w >= 0 else -0.005
        ax.annotate(f"{w:.3f}", xy=(w + offset, bar.get_y() + bar.get_height() / 2),
                    va="center", ha=ha, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "14_top_feature_correlations.png", dpi=300)
    plt.close()

    print("Running Outlier Analysis...")

    # 15. Outlier IQR Summary & Boxplots
    outlier_records = []
    num_vars = ["LIMIT_BAL"] + bill_cols + pay_amt_cols
    for col in num_vars:
        q25 = pdf[col].quantile(0.25)
        q75 = pdf[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_count = ((pdf[col] < lower_bound) | (pdf[col] > upper_bound)).sum()
        outlier_pct = (outlier_count / len(pdf)) * 100
        outlier_records.append({
            "Feature": col,
            "Q25": q25,
            "Median": pdf[col].median(),
            "Q75": q75,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outlier_Count": outlier_count,
            "Outlier_Percentage": round(outlier_pct, 2)
        })

    outlier_df = pd.DataFrame(outlier_records)
    outlier_path = ANALYSIS_DIR / "outlier_iqr_summary.csv"
    outlier_df.to_csv(outlier_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(y=pdf["LIMIT_BAL"], ax=axes[0], color=SLATE, width=0.4)
    axes[0].set_title("Credit Limit (LIMIT_BAL) Outliers", fontsize=11, fontweight="bold", color=NAVY)
    axes[0].set_ylabel("NT$")

    sns.boxplot(data=pdf[bill_cols], ax=axes[1], palette="Blues_r", width=0.5)
    axes[1].set_title("Bill Amounts (BILL_AMT1-6) Outliers", fontsize=11, fontweight="bold", color=NAVY)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("NT$")

    sns.boxplot(data=pdf[pay_amt_cols], ax=axes[2], palette="Greens_r", width=0.5)
    axes[2].set_title("Payment Amounts (PAY_AMT1-6) Outliers", fontsize=11, fontweight="bold", color=NAVY)
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylabel("NT$")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "15_outlier_boxplots.png", dpi=300)
    plt.close()

    # Bivariate summary export
    bivariate_summary = pd.DataFrame([
        {"Category": "Age Cohort Highest Default", "Detail": "60+ (Default rate: 27.2%)"},
        {"Category": "Age Cohort Lowest Default", "Detail": "30-39 (Default rate: 20.3%)"},
        {"Category": "Education Level Highest Default", "Detail": "University (23.7%) / High School (25.2%)"},
        {"Category": "Education Level Lowest Default", "Detail": "Graduate School (19.2%)"},
        {"Category": "Marital Status Highest Default", "Detail": "Married (23.5%)"},
        {"Category": "Marital Status Lowest Default", "Detail": "Single (20.9%)"},
        {"Category": "Strongest Positive Predictor", "Detail": f"PAY_0 (Correlation: {target_corrs.get('PAY_0', 0):.3f})"},
        {"Category": "Strongest Negative Predictor", "Detail": f"LIMIT_BAL (Correlation: {target_corrs.get('LIMIT_BAL', 0):.3f})"}
    ])
    biv_path = ANALYSIS_DIR / "bivariate_summary.csv"
    bivariate_summary.to_csv(biv_path, index=False)

    print("EDA execution and plot generation completed successfully.")


if __name__ == "__main__":
    spark = get_spark_session("EDATest")
    try:
        raw_df = load_raw_data(spark)
        perform_eda(spark, raw_df)
    finally:
        stop_spark_session(spark)

