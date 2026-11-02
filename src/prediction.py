import os
import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Tuple
import pandas as pd
from pyspark.sql import DataFrame, Row, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import IntegerType

from src.utils import (
    BEST_MODEL_PATH,
    PREDICTIONS_DIR,
    DATASET_PATH,
    get_spark_session,
    stop_spark_session,
    ensure_directories,
)
from src.preprocessing import clean_and_standardize_data, engineer_financial_features


class CreditDefaultPredictor:
    """
    Production PySpark Prediction Application for Credit Card Default Risk.
    Loads saved native PipelineModel and performs single and batch inference
    with configurable risk tier thresholds.
    """

    def __init__(self, spark, model_path: Path = BEST_MODEL_PATH):
        self.spark = spark
        self.model_path = model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Saved PySpark model not found at {model_path}. Please train and save the model first.")
        print(f"Loading native PySpark PipelineModel from {model_path}...")
        self.model = PipelineModel.load(str(model_path))
        print("Model loaded successfully.")

    @staticmethod
    def categorize_risk(prob_default: float, low_th: float = 0.25, high_th: float = 0.50) -> Tuple[str, str]:
        """
        Assign understandable risk tier and credit recommendation based on default probability.
        """
        if prob_default < low_th:
            return "Low Risk", "Approved: Low default hazard. Standard credit terms apply."
        elif prob_default < high_th:
            return "Medium Risk", "Conditional: Moderate default probability. Recommend lower credit limit and monitoring."
        else:
            return "High Risk", "Declined: Elevated default risk. Flag for review or collateral requirement."

    def validate_customer_data(self, data: Dict[str, Any]):
        """Validate input payload contains all necessary financial attributes."""
        required_fields = [
            "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
            "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Customer payload missing required fields: {missing}")

    def predict_single(self, customer_data: Dict[str, Any], low_th: float = 0.25, high_th: float = 0.50) -> Dict[str, Any]:
        """
        Execute prediction for a single customer financial profile.
        """
        self.validate_customer_data(customer_data)

        # Convert dictionary to single-row Spark DataFrame
        input_pdf = pd.DataFrame([customer_data])
        raw_df = self.spark.createDataFrame(input_pdf)

        # Preprocess & Feature Engineer
        cleaned_df = clean_and_standardize_data(raw_df)
        fe_df = engineer_financial_features(cleaned_df)

        # Inference through native PySpark PipelineModel
        prediction_df = self.model.transform(fe_df)
        # Use vector_to_array to safely index VectorUDT probability column
        prob_pdf = (
            prediction_df
            .select(
                "prediction",
                vector_to_array(F.col("probability")).getItem(0).alias("p0"),
                vector_to_array(F.col("probability")).getItem(1).alias("p1"),
            )
            .toPandas()
        )

        pred_class = int(prob_pdf["prediction"].iloc[0])
        prob_no_default = float(prob_pdf["p0"].iloc[0])
        prob_default = float(prob_pdf["p1"].iloc[0])

        risk_category, recommendation = self.categorize_risk(prob_default, low_th, high_th)

        return {
            "prediction": pred_class,
            "prediction_label": "Default" if pred_class == 1 else "No Default",
            "probability_default": round(prob_default * 100, 2),
            "probability_no_default": round(prob_no_default * 100, 2),
            "risk_tier": risk_category,
            "recommendation": recommendation,
            "customer_profile": {
                "credit_limit": customer_data.get("LIMIT_BAL"),
                "age": customer_data.get("AGE"),
                "education": customer_data.get("EDUCATION"),
                "marriage": customer_data.get("MARRIAGE"),
                "september_repayment_delay": customer_data.get("PAY_0"),
                "latest_bill": customer_data.get("BILL_AMT1"),
                "latest_payment": customer_data.get("PAY_AMT1"),
            }
        }

    def predict_batch(self, input_csv: Path, output_csv: Path = None, sample_size: int = None,
                      low_th: float = 0.25, high_th: float = 0.50) -> Dict[str, Any]:
        """
        Execute batch prediction for CSV input, generating predictions and probabilities for all records.
        """
        if not input_csv.exists():
            raise FileNotFoundError(f"Input batch CSV not found at {input_csv}")

        raw_df = self.spark.read.csv(str(input_csv), header=True, inferSchema=True)
        for c in raw_df.columns:
            if "." in c:
                raw_df = raw_df.withColumnRenamed(c, c.replace(".", "_"))

        if sample_size and sample_size < raw_df.count():
            raw_df = raw_df.limit(sample_size)

        cleaned_df = clean_and_standardize_data(raw_df)
        fe_df = engineer_financial_features(cleaned_df)

        predictions_df = self.model.transform(fe_df)

        # Use vector_to_array to convert VectorUDT → array before indexing
        result_spark_df = (
            predictions_df
            .withColumn("prob_default", vector_to_array(F.col("probability")).getItem(1))
            .withColumn("prob_no_default", 1.0 - vector_to_array(F.col("probability")).getItem(1))
            .withColumn(
                "Risk_Tier",
                F.when(F.col("prob_default") < low_th, F.lit("Low Risk"))
                .when(F.col("prob_default") < high_th, F.lit("Medium Risk"))
                .otherwise(F.lit("High Risk"))
            )
        )

        select_cols = [
            "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
            "PAY_0", "BILL_AMT1", "PAY_AMT1",
            "label", "prediction", "prob_default", "prob_no_default", "Risk_Tier"
        ]
        available_cols = [c for c in select_cols if c in result_spark_df.columns]
        export_pdf = result_spark_df.select(available_cols).toPandas()

        total_processed = len(export_pdf)
        default_count = int((export_pdf["prediction"] == 1.0).sum())
        non_default_count = total_processed - default_count
        default_pct = round((default_count / total_processed) * 100, 2)

        risk_distribution = export_pdf["Risk_Tier"].value_counts().to_dict()

        if output_csv is None:
            output_csv = PREDICTIONS_DIR / "batch_predictions.csv"

        export_pdf.to_csv(output_csv, index=False)
        print(f"Saved Batch Predictions to: {output_csv}")

        return {
            "total_processed": total_processed,
            "default_predictions": default_count,
            "non_default_predictions": non_default_count,
            "default_percentage": default_pct,
            "risk_distribution": risk_distribution,
            "output_path": str(output_csv),
            "sample_rows": export_pdf.head(10).to_dict(orient="records")
        }


def run_prediction_demo():
    """Demonstrate single and batch prediction execution."""
    ensure_directories()
    spark = get_spark_session("CreditCardPredictionApp")
    try:
        predictor = CreditDefaultPredictor(spark)

        # 1. Sample High-Risk Customer (History of delays)
        sample_high_risk = {
            "LIMIT_BAL": 50000.0, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 42,
            "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 48000.0, "BILL_AMT2": 49000.0, "BILL_AMT3": 49500.0,
            "BILL_AMT4": 45000.0, "BILL_AMT5": 44000.0, "BILL_AMT6": 43000.0,
            "PAY_AMT1": 0.0, "PAY_AMT2": 1000.0, "PAY_AMT3": 0.0,
            "PAY_AMT4": 1500.0, "PAY_AMT5": 1000.0, "PAY_AMT6": 1000.0,
        }

        # 2. Sample Low-Risk Customer (Timely payments, high credit)
        sample_low_risk = {
            "LIMIT_BAL": 250000.0, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 31,
            "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
            "BILL_AMT1": 8000.0, "BILL_AMT2": 5400.0, "BILL_AMT3": 6200.0,
            "BILL_AMT4": 3500.0, "BILL_AMT5": 4100.0, "BILL_AMT6": 2900.0,
            "PAY_AMT1": 8000.0, "PAY_AMT2": 5400.0, "PAY_AMT3": 6200.0,
            "PAY_AMT4": 3500.0, "PAY_AMT5": 4100.0, "PAY_AMT6": 2900.0,
        }

        print("\n" + "=" * 60)
        print("SINGLE PREDICTION DEMONSTRATION")
        print("=" * 60)
        print("\nEvaluating Profile A (Delinquent History):")
        res_a = predictor.predict_single(sample_high_risk)
        print(f"Prediction: {res_a['prediction_label']} (Class {res_a['prediction']})")
        print(f"Default Probability: {res_a['probability_default']}% | Non-Default: {res_a['probability_no_default']}%")
        print(f"Risk Tier: {res_a['risk_tier']}")
        print(f"Recommendation: {res_a['recommendation']}")

        print("\nEvaluating Profile B (Prime Customer):")
        res_b = predictor.predict_single(sample_low_risk)
        print(f"Prediction: {res_b['prediction_label']} (Class {res_b['prediction']})")
        print(f"Default Probability: {res_b['probability_default']}% | Non-Default: {res_b['probability_no_default']}%")
        print(f"Risk Tier: {res_b['risk_tier']}")
        print(f"Recommendation: {res_b['recommendation']}")

        print("\n" + "=" * 60)
        print("BATCH PREDICTION DEMONSTRATION")
        print("=" * 60)
        batch_res = predictor.predict_batch(DATASET_PATH, sample_size=1000)
        print(f"Total Processed: {batch_res['total_processed']}")
        print(f"Predicted Defaults: {batch_res['default_predictions']} ({batch_res['default_percentage']}%)")
        print(f"Risk Distribution: {batch_res['risk_distribution']}")

    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Default Risk Prediction App")
    parser.add_argument("--mode", choices=["single", "batch", "demo"], default="demo", help="Prediction execution mode")
    parser.add_argument("--input", type=str, default=str(DATASET_PATH), help="Input CSV path for batch prediction")
    parser.add_argument("--output", type=str, default=str(PREDICTIONS_DIR / "batch_predictions.csv"), help="Output CSV path")
    args = parser.parse_args()

    if args.mode == "demo":
        run_prediction_demo()
    elif args.mode == "batch":
        spark = get_spark_session("BatchPrediction")
        try:
            predictor = CreditDefaultPredictor(spark)
            predictor.predict_batch(Path(args.input), Path(args.output))
        finally:
            stop_spark_session(spark)

