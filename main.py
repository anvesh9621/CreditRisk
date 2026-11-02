import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (
    get_spark_session,
    stop_spark_session,
    ensure_directories,
    DATASET_PATH,
    PREDICTIONS_DIR,
    timer,
)
from src.data_loader import load_raw_data, explore_dataset
from src.eda import perform_eda
from src.preprocessing import (
    clean_and_standardize_data,
    engineer_financial_features,
    split_data,
    export_preprocessing_summary,
)
from src.model_training import train_and_evaluate_all_models
from src.model_comparison import compare_and_select_best_model
from src.prediction import CreditDefaultPredictor
from src.dashboard_exporter import export_dashboard_data


def run_pipeline():
    """
    Execute the entire Machine Learning Pipeline for
    Predicting Credit Card Default Using Apache Spark and PySpark MLlib.
    """
    total_start = time.time()
    print("=" * 80)
    print("PREDICTING CREDIT CARD DEFAULT USING APACHE SPARK & PYSPARK ML")
    print("=" * 80)

    ensure_directories()
    spark = get_spark_session("CreditCardDefaultPipeline")

    try:
        # Step 1: Data Loading & Exploration
        with timer("Step 1: Dataset Loading & Schema Exploration"):
            raw_df = load_raw_data(spark, DATASET_PATH)
            exploration_stats = explore_dataset(spark, raw_df)

        # Step 2: Exploratory Data Analysis & Visualization
        with timer("Step 2: Exploratory Data Analysis (EDA)"):
            perform_eda(spark, raw_df)

        # Step 3: Data Preprocessing & Feature Engineering
        with timer("Step 3: Preprocessing & Financial Feature Engineering"):
            clean_df = clean_and_standardize_data(raw_df)
            fe_df = engineer_financial_features(clean_df)
            train_df, test_df = split_data(fe_df, train_ratio=0.8, seed=42)
            export_preprocessing_summary()

        # Step 4: Model Training, Tuning, and Imbalance Handling
        with timer("Step 4: Model Training & Hyperparameter Tuning"):
            training_results = train_and_evaluate_all_models(train_df, test_df)

        # Step 5: Model Comparison & Best Model Saving
        with timer("Step 5: Model Comparison, Selection & Best Model Persistence"):
            comparison_results = compare_and_select_best_model(training_results)

        # Step 6: Prediction Application Execution (Single & Batch)
        with timer("Step 6: Production Inference Execution"):
            predictor = CreditDefaultPredictor(spark)

            # High Risk Test Profile
            profile_high = {
                "LIMIT_BAL": 30000.0, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 40,
                "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 28000.0, "BILL_AMT2": 29000.0, "BILL_AMT3": 29500.0,
                "BILL_AMT4": 27000.0, "BILL_AMT5": 26000.0, "BILL_AMT6": 25000.0,
                "PAY_AMT1": 0.0, "PAY_AMT2": 500.0, "PAY_AMT3": 0.0,
                "PAY_AMT4": 1000.0, "PAY_AMT5": 500.0, "PAY_AMT6": 500.0,
            }
            pred_high = predictor.predict_single(profile_high)
            print(f"\n[Inference Profile - High Risk]: Class {pred_high['prediction']} ({pred_high['prediction_label']}) | Default Prob: {pred_high['probability_default']}% | Tier: {pred_high['risk_tier']}")

            # Low Risk Test Profile
            profile_low = {
                "LIMIT_BAL": 200000.0, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 32,
                "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
                "BILL_AMT1": 5000.0, "BILL_AMT2": 4500.0, "BILL_AMT3": 4000.0,
                "BILL_AMT4": 3000.0, "BILL_AMT5": 2500.0, "BILL_AMT6": 2000.0,
                "PAY_AMT1": 5000.0, "PAY_AMT2": 4500.0, "PAY_AMT3": 4000.0,
                "PAY_AMT4": 3000.0, "PAY_AMT5": 2500.0, "PAY_AMT6": 2000.0,
            }
            pred_low = predictor.predict_single(profile_low)
            print(f"[Inference Profile - Low Risk]:  Class {pred_low['prediction']} ({pred_low['prediction_label']}) | Default Prob: {pred_low['probability_default']}% | Tier: {pred_low['risk_tier']}")

            # Batch Prediction on Test Set / Dataset Sample
            batch_csv_out = PREDICTIONS_DIR / "batch_predictions.csv"
            batch_results = predictor.predict_batch(DATASET_PATH, output_csv=batch_csv_out, sample_size=3000)
            print(f"\n[Batch Inference]: Processed {batch_results['total_processed']} records. Predicted {batch_results['default_predictions']} defaults ({batch_results['default_percentage']}%).")

        # Step 7: Export Dashboard Data
        with timer("Step 7: Dashboard Data Packaging"):
            export_dashboard_data(comparison_results, batch_results)

        total_elapsed = time.time() - total_start
        print("\n" + "=" * 80)
        print(f"PIPELINE EXECUTED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS ({total_elapsed/60:.2f} MINUTES)")
        print(f"Best Model: {comparison_results['best_model_name']}")
        print(f"Saved Model Path: models/best_model")
        print(f"Outputs Directory: outputs/")
        print("=" * 80)

    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    run_pipeline()

