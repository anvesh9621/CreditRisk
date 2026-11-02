import time
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)


def _extract_prob_pandas(predictions: DataFrame) -> pd.DataFrame:
    """
    Extract label and probability-of-default (class 1) into a Pandas DataFrame.
    VectorUDT must be converted to a plain array via vector_to_array() before indexing.
    """
    return (
        predictions
        .select(
            F.col("label").cast(DoubleType()).alias("label"),
            vector_to_array(F.col("probability")).getItem(1)
              .cast(DoubleType()).alias("prob_default"),
        )
        .toPandas()
    )


def evaluate_predictions(
    predictions: DataFrame, model_name: str, training_time: float = 0.0
) -> dict:
    """
    Compute comprehensive classification metrics from test-set predictions.
    """
    t0 = time.time()
    predictions = predictions.cache()
    _ = predictions.count()          # materialise cache
    prediction_time = round(time.time() - t0, 3)

    # ── Standard evaluators ─────────────────────────────────────────────────
    def mc_eval(metric, **kw):
        return float(
            MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction",
                metricName=metric, **kw
            ).evaluate(predictions)
        )

    def bin_eval(metric):
        return float(
            BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction",
                metricName=metric
            ).evaluate(predictions)
        )

    accuracy           = mc_eval("accuracy")
    precision_weighted = mc_eval("weightedPrecision")
    recall_weighted    = mc_eval("weightedRecall")
    f1_weighted        = mc_eval("f1")
    precision_c1       = mc_eval("precisionByLabel",  metricLabel=1.0)
    recall_c1          = mc_eval("recallByLabel",     metricLabel=1.0)
    f1_c1              = mc_eval("fMeasureByLabel",   metricLabel=1.0)
    roc_auc            = bin_eval("areaUnderROC")
    pr_auc             = bin_eval("areaUnderPR")

    # ── Confusion-matrix counts ──────────────────────────────────────────────
    tp = predictions.filter((F.col("label") == 1.0) & (F.col("prediction") == 1.0)).count()
    fp = predictions.filter((F.col("label") == 0.0) & (F.col("prediction") == 1.0)).count()
    tn = predictions.filter((F.col("label") == 0.0) & (F.col("prediction") == 0.0)).count()
    fn = predictions.filter((F.col("label") == 1.0) & (F.col("prediction") == 0.0)).count()
    total = tp + fp + tn + fn

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ── Composite validation score ───────────────────────────────────────────
    # 0.35*ROC-AUC + 0.25*PR-AUC + 0.25*Recall_Default + 0.15*F1_Default
    composite_score = 0.35 * roc_auc + 0.25 * pr_auc + 0.25 * recall_c1 + 0.15 * f1_c1

    return {
        "model_name":        model_name,
        "accuracy":          round(accuracy, 4),
        "precision":         round(precision_weighted, 4),
        "recall":            round(recall_weighted, 4),
        "f1_score":          round(f1_weighted, 4),
        "precision_default": round(precision_c1, 4),
        "recall_default":    round(recall_c1, 4),
        "f1_default":        round(f1_c1, 4),
        "roc_auc":           round(roc_auc, 4),
        "pr_auc":            round(pr_auc, 4),
        "specificity":       round(specificity, 4),
        "training_time":     round(training_time, 2),
        "prediction_time":   prediction_time,
        "confusion_matrix": {
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "total": int(total),
        },
        "composite_score": round(composite_score, 4),
    }


def compute_roc_pr_curve_data(
    predictions: DataFrame, num_thresholds: int = 50
) -> dict:
    """
    Compute ROC and Precision-Recall curve coordinates.
    Uses toPandas() (no RDD ops) to avoid Python-worker crashes on Windows.
    """
    pdf = _extract_prob_pandas(predictions)
    labels = pdf["label"].values
    scores = pdf["prob_default"].values

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    p_total = np.sum(labels == 1.0)
    n_total = np.sum(labels == 0.0)

    roc_points, pr_points = [], []
    for th in thresholds:
        preds = (scores >= th).astype(float)
        tp = np.sum((labels == 1.0) & (preds == 1.0))
        fp = np.sum((labels == 0.0) & (preds == 1.0))
        fn = np.sum((labels == 1.0) & (preds == 0.0))

        tpr  = tp / p_total      if p_total > 0            else 0.0
        fpr  = fp / n_total      if n_total > 0            else 0.0
        prec = tp / (tp + fp)    if (tp + fp) > 0          else 1.0

        roc_points.append({"threshold": round(float(th), 3),
                            "fpr": round(float(fpr), 4),
                            "tpr": round(float(tpr), 4)})
        pr_points.append({"threshold": round(float(th), 3),
                           "precision": round(float(prec), 4),
                           "recall": round(float(tpr), 4)})

    return {"roc_curve": roc_points, "pr_curve": pr_points}
