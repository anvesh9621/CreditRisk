import time
from typing import Dict, Tuple
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    MinMaxScaler,
)
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
    NaiveBayes,
)
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from src.model_evaluation import evaluate_predictions, compute_roc_pr_curve_data
from src.preprocessing import add_class_weights


def build_pipeline_stages():
    """
    Build shared PySpark ML preprocessing pipeline stages.
    Returns three assembler paths:
      - stages_scaled: for linear models (LR) using StandardScaler
      - stages_tree:   for tree models using raw features
      - stages_nb:     for Naive Bayes using MinMaxScaler (non-negative)
    """
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec")
        for col in categorical_cols
    ]

    base_num_cols = (
        ["LIMIT_BAL", "AGE"]
        + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        + [f"BILL_AMT{i}" for i in range(1, 7)]
        + [f"PAY_AMT{i}" for i in range(1, 7)]
    )

    engineered_cols = [
        "TOTAL_BILL_AMT", "AVG_BILL_AMT", "TOTAL_PAY_AMT", "AVG_PAY_AMT",
        "MAX_BILL_AMT", "MIN_BILL_AMT", "PAY_TO_BILL_RATIO",
        "DELAYED_MONTHS_COUNT", "MAX_REPAYMENT_DELAY", "AVG_REPAYMENT_STATUS",
        "CREDIT_UTILIZATION_RATIO", "PAYMENT_CONSISTENCY",
    ]

    encoded_cat_cols = [f"{col}_vec" for col in categorical_cols]
    all_input_cols = encoded_cat_cols + base_num_cols + engineered_cols
    feature_names = categorical_cols + base_num_cols + engineered_cols

    assembler = VectorAssembler(
        inputCols=all_input_cols, outputCol="features", handleInvalid="skip"
    )

    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withStd=True, withMean=False
    )

    # Assembler for Naive Bayes: strictly non-negative categorical & count features
    nb_assembler = VectorAssembler(
        inputCols=encoded_cat_cols + ["DELAYED_MONTHS_COUNT"],
        outputCol="nb_features",
        handleInvalid="skip"
    )

    # All models share indexers + encoders + assembler
    # Tree models stop here (use "features"); LR adds StandardScaler; NB uses nb_features
    common = indexers + encoders + [assembler]
    scaled_stages = common + [scaler]
    nb_stages = indexers + encoders + [nb_assembler]

    return scaled_stages, common, nb_stages, feature_names


def _tvs_fit(pipeline, param_grid, train_df):
    """Run TrainValidationSplit and return the best fitted PipelineModel."""
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        trainRatio=0.8,
        seed=42,
    )
    return tvs.fit(train_df).bestModel


def train_logistic_regression(
    train_df: DataFrame, scaled_stages: list, use_class_weights: bool = True
) -> Tuple[PipelineModel, float, dict]:
    """Train and tune Logistic Regression (baseline model)."""
    tag = "Weighted" if use_class_weights else "Unweighted"
    print(f"\n--- Training Logistic Regression ({tag}) ---")
    t0 = time.time()

    # Build LR with or without class weighting — never pass None to weightCol
    if use_class_weights:
        lr = LogisticRegression(
            featuresCol="scaled_features", labelCol="label",
            maxIter=50, weightCol="class_weight"
        )
    else:
        lr = LogisticRegression(
            featuresCol="scaled_features", labelCol="label", maxIter=50
        )

    pipeline = Pipeline(stages=scaled_stages + [lr])
    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )
    best = _tvs_fit(pipeline, param_grid, train_df)
    elapsed = time.time() - t0

    best_lr = best.stages[-1]
    params = {
        "regParam": float(best_lr.getOrDefault("regParam")),
        "elasticNetParam": float(best_lr.getOrDefault("elasticNetParam")),
        "use_class_weights": use_class_weights,
    }
    return best, elapsed, params


def train_decision_tree(
    train_df: DataFrame, tree_stages: list
) -> Tuple[PipelineModel, float, dict]:
    """Train and tune Decision Tree classifier."""
    print("\n--- Training Decision Tree Classifier ---")
    t0 = time.time()

    dt = DecisionTreeClassifier(
        featuresCol="features", labelCol="label", seed=42
    )
    pipeline = Pipeline(stages=tree_stages + [dt])
    param_grid = (
        ParamGridBuilder()
        .addGrid(dt.maxDepth, [5, 8])
        .addGrid(dt.minInstancesPerNode, [10, 30])
        .build()
    )
    best = _tvs_fit(pipeline, param_grid, train_df)
    elapsed = time.time() - t0

    best_dt = best.stages[-1]
    params = {
        "maxDepth": int(best_dt.getOrDefault("maxDepth")),
        "minInstancesPerNode": int(best_dt.getOrDefault("minInstancesPerNode")),
    }
    return best, elapsed, params


def train_random_forest(
    train_df: DataFrame, tree_stages: list
) -> Tuple[PipelineModel, float, dict]:
    """Train and tune Random Forest classifier."""
    print("\n--- Training Random Forest Classifier ---")
    t0 = time.time()

    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label", seed=42
    )
    pipeline = Pipeline(stages=tree_stages + [rf])
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [30, 50])
        .addGrid(rf.maxDepth, [6, 10])
        .build()
    )
    best = _tvs_fit(pipeline, param_grid, train_df)
    elapsed = time.time() - t0

    best_rf = best.stages[-1]
    params = {
        "numTrees": int(best_rf.getOrDefault("numTrees")),
        "maxDepth": int(best_rf.getOrDefault("maxDepth")),
    }
    return best, elapsed, params


def train_gradient_boosted_tree(
    train_df: DataFrame, tree_stages: list
) -> Tuple[PipelineModel, float, dict]:
    """Train and tune Gradient Boosted Tree (GBT) classifier."""
    print("\n--- Training Gradient Boosted Tree Classifier ---")
    t0 = time.time()

    gbt = GBTClassifier(
        featuresCol="features", labelCol="label", seed=42
    )
    pipeline = Pipeline(stages=tree_stages + [gbt])
    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [25, 40])
        .addGrid(gbt.maxDepth, [4, 6])
        .build()
    )
    best = _tvs_fit(pipeline, param_grid, train_df)
    elapsed = time.time() - t0

    best_gbt = best.stages[-1]
    params = {
        "maxIter": int(best_gbt.getOrDefault("maxIter")),
        "maxDepth": int(best_gbt.getOrDefault("maxDepth")),
    }
    return best, elapsed, params


def train_naive_bayes(
    train_df: DataFrame, nb_stages: list
) -> Tuple[PipelineModel, float, dict]:
    """Train and tune Naive Bayes on non-negative MinMax-scaled features."""
    print("\n--- Training Naive Bayes Classifier ---")
    t0 = time.time()

    nb = NaiveBayes(
        featuresCol="nb_features", labelCol="label", modelType="multinomial"
    )
    pipeline = Pipeline(stages=nb_stages + [nb])
    param_grid = (
        ParamGridBuilder()
        .addGrid(nb.smoothing, [0.5, 1.0])
        .build()
    )
    best = _tvs_fit(pipeline, param_grid, train_df)
    elapsed = time.time() - t0

    best_nb = best.stages[-1]
    params = {
        "smoothing": float(best_nb.getOrDefault("smoothing")),
        "modelType": str(best_nb.getOrDefault("modelType")),
    }
    return best, elapsed, params


def train_and_evaluate_all_models(
    train_df: DataFrame, test_df: DataFrame
) -> dict:
    """
    Train, tune, and evaluate all 5 PySpark MLlib classification models.
    Includes a baseline unweighted Logistic Regression to demonstrate
    class imbalance handling impact.
    """
    # Prepare class-weighted training set once
    train_weighted = add_class_weights(train_df)

    # Build pipeline stage sets once
    scaled_stages, tree_stages, nb_stages, feature_names = build_pipeline_stages()

    results = {}
    models = {}
    curve_data = {}
    params_dict = {}
    test_preds = {}

    # Helper to register a trained model
    def register(name, model, train_time, params, preds):
        metrics = evaluate_predictions(preds, name, training_time=train_time)
        curves = compute_roc_pr_curve_data(preds)
        results[name] = metrics
        models[name] = model
        curve_data[name] = curves
        params_dict[name] = params
        test_preds[name] = preds

    # 1. Baseline Logistic Regression (no class weights)
    m, t, p = train_logistic_regression(train_weighted, list(scaled_stages), use_class_weights=False)
    preds = m.transform(test_df)
    register("Logistic Regression (Unweighted)", m, t, p, preds)

    # 2. Logistic Regression (class-weighted)
    m, t, p = train_logistic_regression(train_weighted, list(scaled_stages), use_class_weights=True)
    preds = m.transform(test_df)
    register("Logistic Regression (Weighted)", m, t, p, preds)

    # 3. Decision Tree
    m, t, p = train_decision_tree(train_weighted, list(tree_stages))
    preds = m.transform(test_df)
    register("Decision Tree", m, t, p, preds)

    # 4. Random Forest
    m, t, p = train_random_forest(train_weighted, list(tree_stages))
    preds = m.transform(test_df)
    register("Random Forest", m, t, p, preds)

    # 5. Gradient Boosted Tree
    m, t, p = train_gradient_boosted_tree(train_weighted, list(tree_stages))
    preds = m.transform(test_df)
    register("Gradient Boosted Tree", m, t, p, preds)

    # 6. Naive Bayes
    m, t, p = train_naive_bayes(train_weighted, list(nb_stages))
    preds = m.transform(test_df)
    register("Naive Bayes", m, t, p, preds)

    return {
        "metrics": results,
        "models": models,
        "curve_data": curve_data,
        "hyperparameters": params_dict,
        "feature_names": feature_names,
        "test_predictions": test_preds,
    }
