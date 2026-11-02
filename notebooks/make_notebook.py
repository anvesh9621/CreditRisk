import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Predicting Credit Card Default Using Apache Spark and PySpark MLlib\n",
                "\n",
                "## 1. Project Objective\n",
                "Build an end-to-end distributed Machine Learning pipeline using **Apache Spark** and **PySpark MLlib** to predict whether credit card clients will default on their next payment.\n",
                "\n",
                "### Target Variable:\n",
                "- `0`: No Default (Repaid on time)\n",
                "- `1`: Default (Delinquent payment)\n",
                "\n",
                "### Key Constraints:\n",
                "- Pure Apache Spark & PySpark MLlib throughout (no Scikit-learn replacement)\n",
                "- Class imbalance handling (~3.5:1 ratio)\n",
                "- Hyperparameter tuning using `TrainValidationSplit` and `ParamGridBuilder`\n",
                "- Multi-criteria evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "from pathlib import Path\n",
                "sys.path.insert(0, str(Path.cwd().parent))\n",
                "\n",
                "from src.utils import get_spark_session, stop_spark_session, DATASET_PATH, BEST_MODEL_PATH\n",
                "from src.data_loader import load_raw_data, explore_dataset\n",
                "from src.preprocessing import clean_and_standardize_data, engineer_financial_features, split_data, add_class_weights\n",
                "from src.model_training import train_and_evaluate_all_models\n",
                "from src.model_comparison import compare_and_select_best_model\n",
                "from src.prediction import CreditDefaultPredictor\n",
                "\n",
                "spark = get_spark_session('CreditDefaultNotebook')\n",
                "print('Spark Version:', spark.version)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Loading & Exploration\n",
                "Loading the UCI Credit Card Clients dataset (30,000 records, 24 features)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_df = load_raw_data(spark, DATASET_PATH)\n",
                "print('Total Records:', raw_df.count(), '| Total Columns:', len(raw_df.columns))\n",
                "raw_df.printSchema()\n",
                "raw_df.show(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Preprocessing & Financial Feature Engineering\n",
                "Cleaning anomalies, standardizing demographic codes, and engineering 11 domain features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "cleaned_df = clean_and_standardize_data(raw_df)\n",
                "fe_df = engineer_financial_features(cleaned_df)\n",
                "train_df, test_df = split_data(fe_df, train_ratio=0.8, seed=42)\n",
                "print('Feature Engineered Schema:')\n",
                "fe_df.select('LIMIT_BAL', 'AGE', 'TOTAL_BILL_AMT', 'PAY_TO_BILL_RATIO', 'CREDIT_UTILIZATION_RATIO', 'DELAYED_MONTHS_COUNT', 'label').show(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Training, Tuning, and Comparison\n",
                "Training 6 model architectures with hyperparameter grids via TrainValidationSplit:\n",
                "1. Logistic Regression (Unweighted Baseline)\n",
                "2. Logistic Regression (Class Weighted)\n",
                "3. Decision Tree Classifier\n",
                "4. Random Forest Classifier\n",
                "5. Gradient Boosted Tree (GBT) Classifier\n",
                "6. Multinomial Naive Bayes"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "training_results = train_and_evaluate_all_models(train_df, test_df)\n",
                "comparison_results = compare_and_select_best_model(training_results)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Model Comparison Summary Table\n",
                "Inspect the ranked performance metrics across all models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "comp_df = comparison_results['comparison_df']\n",
                "display_cols = ['Model', 'Accuracy', 'Default Recall', 'F1-score', 'ROC-AUC', 'PR-AUC', 'Training Time (s)']\n",
                "comp_df[display_cols]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Real-Time Customer Inference Demonstration\n",
                "Loading the native saved PySpark `PipelineModel` and evaluating individual credit profiles."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [],
            "source": [
                "predictor = CreditDefaultPredictor(spark, BEST_MODEL_PATH)\n",
                "\n",
                "# Delinquent Customer Test Profile\n",
                "delinquent_profile = {\n",
                "    'LIMIT_BAL': 40000.0, 'SEX': 1, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 38,\n",
                "    'PAY_0': 2, 'PAY_2': 2, 'PAY_3': 2, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,\n",
                "    'BILL_AMT1': 35000.0, 'BILL_AMT2': 36000.0, 'BILL_AMT3': 36500.0,\n",
                "    'BILL_AMT4': 30000.0, 'BILL_AMT5': 29000.0, 'BILL_AMT6': 28000.0,\n",
                "    'PAY_AMT1': 0.0, 'PAY_AMT2': 1000.0, 'PAY_AMT3': 0.0,\n",
                "    'PAY_AMT4': 1000.0, 'PAY_AMT5': 500.0, 'PAY_AMT6': 500.0,\n",
                "}\n",
                "pred_res = predictor.predict_single(delinquent_profile)\n",
                "print(f\"Prediction: {pred_res['prediction_label']} (Class {pred_res['prediction']})\")\n",
                "print(f\"Default Probability: {pred_res['probability_default']}%\")\n",
                "print(f\"Assigned Risk Tier: {pred_res['risk_tier']}\")\n",
                "print(f\"Decisioning: {pred_res['recommendation']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Clean Shutdown\n",
                "Stop the active SparkSession."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {},
            "outputs": [],
            "source": [
                "stop_spark_session(spark)\n",
                "print('SparkSession stopped cleanly.')"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebooks/credit_card_default_analysis.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print(f"Generated Jupyter Notebook: {out_path}")

