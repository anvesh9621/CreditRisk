"""
Resume training from Random Forest onwards.
LR (unweighted), LR (weighted), and Decision Tree already completed.
Loads their saved predictions from Parquet cache, then trains RF, GBT, NB.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import get_spark_session, stop_spark_session, ensure_directories, METRICS_DIR, PREDICTIONS_DIR, DATASET_PATH, timer
from src.data_loader import load_raw_data
from src.preprocessing import clean_and_standardize_data, engineer_financial_features, split_data, add_class_weights, export_preprocessing_summary
from src.model_training import build_pipeline_stages, train_random_forest, train_gradient_boosted_tree, train_naive_bayes, train_logistic_regression, train_decision_tree
from src.model_evaluation import evaluate_predictions, compute_roc_pr_curve_data
from src.model_comparison import compare_and_select_best_model
from src.prediction import CreditDefaultPredictor
from src.dashboard_exporter import export_dashboard_data
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier


def run_resume():
    ensure_directories()
    spark = get_spark_session("CreditCardDefaultPipeline")
    try:
        raw_df = load_raw_data(spark)
        clean_df = clean_and_standardize_data(raw_df)
        fe_df = engineer_financial_features(clean_df)
        train_df, test_df = split_data(fe_df, train_ratio=0.8, seed=42)
        train_weighted = add_class_weights(train_df)

        scaled_stages, tree_stages, nb_stages, feature_names = build_pipeline_stages()

        results = {}
        models = {}
        curve_data = {}
        params_dict = {}
        test_preds = {}

        def register(name, model, train_time, params, preds):
            metrics = evaluate_predictions(preds, name, training_time=train_time)
            curves = compute_roc_pr_curve_data(preds)
            results[name] = metrics
            models[name] = model
            curve_data[name] = curves
            params_dict[name] = params
            test_preds[name] = preds
            print(f"[{name}] Acc={metrics['accuracy']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | Recall(Default)={metrics['recall_default']:.4f}")

        print("\n[Step 4] Training all 6 models...")

        # 1. LR Unweighted
        with timer("LR Unweighted"):
            m, t, p = train_logistic_regression(train_weighted, list(scaled_stages), use_class_weights=False)
            preds = m.transform(test_df)
            register("Logistic Regression (Unweighted)", m, t, p, preds)

        # 2. LR Weighted
        with timer("LR Weighted"):
            m, t, p = train_logistic_regression(train_weighted, list(scaled_stages), use_class_weights=True)
            preds = m.transform(test_df)
            register("Logistic Regression (Weighted)", m, t, p, preds)

        # 3. Decision Tree
        with timer("Decision Tree"):
            m, t, p = train_decision_tree(train_weighted, list(tree_stages))
            preds = m.transform(test_df)
            register("Decision Tree", m, t, p, preds)

        # 4. Random Forest
        with timer("Random Forest"):
            m, t, p = train_random_forest(train_weighted, list(tree_stages))
            preds = m.transform(test_df)
            register("Random Forest", m, t, p, preds)

        # 5. GBT
        with timer("GBT"):
            m, t, p = train_gradient_boosted_tree(train_weighted, list(tree_stages))
            preds = m.transform(test_df)
            register("Gradient Boosted Tree", m, t, p, preds)

        # 6. Naive Bayes
        with timer("Naive Bayes"):
            m, t, p = train_naive_bayes(train_weighted, list(nb_stages))
            preds = m.transform(test_df)
            register("Naive Bayes", m, t, p, preds)

        training_results = {
            "metrics": results, "models": models, "curve_data": curve_data,
            "hyperparameters": params_dict, "feature_names": feature_names, "test_predictions": test_preds
        }

        # Step 5: Compare & save
        with timer("Model Comparison & Best Model Saving"):
            comparison_results = compare_and_select_best_model(training_results)

        # Step 6: Predictions
        with timer("Inference"):
            predictor = CreditDefaultPredictor(spark)
            profile_high = {
                "LIMIT_BAL": 30000.0, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 40,
                "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 28000.0, "BILL_AMT2": 29000.0, "BILL_AMT3": 29500.0,
                "BILL_AMT4": 27000.0, "BILL_AMT5": 26000.0, "BILL_AMT6": 25000.0,
                "PAY_AMT1": 0.0, "PAY_AMT2": 500.0, "PAY_AMT3": 0.0,
                "PAY_AMT4": 1000.0, "PAY_AMT5": 500.0, "PAY_AMT6": 500.0,
            }
            profile_low = {
                "LIMIT_BAL": 200000.0, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 32,
                "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
                "BILL_AMT1": 5000.0, "BILL_AMT2": 4500.0, "BILL_AMT3": 4000.0,
                "BILL_AMT4": 3000.0, "BILL_AMT5": 2500.0, "BILL_AMT6": 2000.0,
                "PAY_AMT1": 5000.0, "PAY_AMT2": 4500.0, "PAY_AMT3": 4000.0,
                "PAY_AMT4": 3000.0, "PAY_AMT5": 2500.0, "PAY_AMT6": 2000.0,
            }
            res_h = predictor.predict_single(profile_high)
            res_l = predictor.predict_single(profile_low)
            print(f"High-Risk Profile: {res_h['prediction_label']} | Default Prob: {res_h['probability_default']}% | Tier: {res_h['risk_tier']}")
            print(f"Low-Risk Profile:  {res_l['prediction_label']} | Default Prob: {res_l['probability_default']}% | Tier: {res_l['risk_tier']}")
            batch_results = predictor.predict_batch(DATASET_PATH, sample_size=3000)
            print(f"Batch: {batch_results['total_processed']} records | {batch_results['default_predictions']} defaults ({batch_results['default_percentage']}%)")

        # Step 7: Dashboard export
        with timer("Dashboard Data Export"):
            export_dashboard_data(comparison_results, batch_results)

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print(f"Best Model: {comparison_results['best_model_name']}")
        print("="*70)

    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    run_resume()

