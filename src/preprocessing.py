import pandas as pd
from pathlib import Path
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
)

from src.utils import (
    ANALYSIS_DIR,
    get_spark_session,
    stop_spark_session,
    ensure_directories,
)
from src.data_loader import load_raw_data


def clean_and_standardize_data(df: DataFrame) -> DataFrame:
    """
    1. Drop ID column.
    2. Handle invalid/unassigned categorical codes in EDUCATION and MARRIAGE:
       - EDUCATION: 0, 4, 5, 6 -> 4 (Others/Unknown)
       - MARRIAGE: 0, 3 -> 3 (Others/Unknown)
    3. Ensure target column is named 'label' (DoubleType).
    """
    # Drop ID
    if "ID" in df.columns:
        df = df.drop("ID")

    # Target column standardization (if present)
    for target_candidate in ["default_payment_next_month", "default.payment.next.month", "label"]:
        if target_candidate in df.columns:
            if target_candidate != "label":
                df = df.withColumn("label", F.col(target_candidate).cast(DoubleType()))
                df = df.drop(target_candidate)
            else:
                df = df.withColumn("label", F.col("label").cast(DoubleType()))
            break

    # Standardize EDUCATION: 1=Grad School, 2=University, 3=High School, 4=Others
    df = df.withColumn(
        "EDUCATION",
        F.when(F.col("EDUCATION").isin([1, 2, 3]), F.col("EDUCATION"))
        .otherwise(F.lit(4))
        .cast(IntegerType())
    )

    # Standardize MARRIAGE: 1=Married, 2=Single, 3=Others
    df = df.withColumn(
        "MARRIAGE",
        F.when(F.col("MARRIAGE").isin([1, 2]), F.col("MARRIAGE"))
        .otherwise(F.lit(3))
        .cast(IntegerType())
    )

    # Ensure SEX is IntegerType
    df = df.withColumn("SEX", F.col("SEX").cast(IntegerType()))

    return df


def engineer_financial_features(df: DataFrame) -> DataFrame:
    """
    Engineer domain-specific financial features:
    - AVG_BILL_AMT & TOTAL_BILL_AMT
    - AVG_PAY_AMT & TOTAL_PAY_AMT
    - MAX_BILL_AMT & MIN_BILL_AMT
    - PAY_TO_BILL_RATIO
    - DELAYED_MONTHS_COUNT
    - MAX_REPAYMENT_DELAY
    - AVG_REPAYMENT_STATUS
    - CREDIT_UTILIZATION_RATIO
    - PAYMENT_CONSISTENCY
    """
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    repay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    # 1. Total & Average Bill Amounts
    total_bill_expr = sum(F.col(c) for c in bill_cols)
    df = df.withColumn("TOTAL_BILL_AMT", total_bill_expr)
    df = df.withColumn("AVG_BILL_AMT", total_bill_expr / 6.0)

    # 2. Total & Average Payment Amounts
    total_pay_expr = sum(F.col(c) for c in pay_cols)
    df = df.withColumn("TOTAL_PAY_AMT", total_pay_expr)
    df = df.withColumn("AVG_PAY_AMT", total_pay_expr / 6.0)

    # 3. Max & Min Bill Amounts
    df = df.withColumn("MAX_BILL_AMT", F.greatest(*[F.col(c) for c in bill_cols]))
    df = df.withColumn("MIN_BILL_AMT", F.least(*[F.col(c) for c in bill_cols]))

    # 4. Payment to Bill Ratio (handled division by zero and bounded between 0 and 5)
    raw_pay_ratio = F.when(F.col("TOTAL_BILL_AMT") <= 0, F.lit(1.0)).otherwise(
        F.col("TOTAL_PAY_AMT") / (F.col("TOTAL_BILL_AMT") + 1e-5)
    )
    df = df.withColumn(
        "PAY_TO_BILL_RATIO",
        F.when(raw_pay_ratio < 0.0, F.lit(0.0))
        .when(raw_pay_ratio > 5.0, F.lit(5.0))
        .otherwise(raw_pay_ratio)
    )

    # 5. Number of delayed-payment months (PAY_i > 0)
    delayed_months_expr = sum(
        F.when(F.col(c) > 0, F.lit(1)).otherwise(F.lit(0)) for c in repay_cols
    )
    df = df.withColumn("DELAYED_MONTHS_COUNT", delayed_months_expr)

    # 6. Maximum repayment delay
    df = df.withColumn("MAX_REPAYMENT_DELAY", F.greatest(*[F.col(c) for c in repay_cols]))

    # 7. Average repayment status
    avg_repay_expr = sum(F.col(c) for c in repay_cols) / 6.0
    df = df.withColumn("AVG_REPAYMENT_STATUS", avg_repay_expr)

    # 8. Credit utilization ratio (BILL_AMT1 / LIMIT_BAL)
    raw_util = F.when(F.col("LIMIT_BAL") <= 0, F.lit(0.0)).otherwise(
        F.col("BILL_AMT1") / F.col("LIMIT_BAL")
    )
    df = df.withColumn(
        "CREDIT_UTILIZATION_RATIO",
        F.when(raw_util < -1.0, F.lit(-1.0))
        .when(raw_util > 5.0, F.lit(5.0))
        .otherwise(raw_util)
    )

    # 9. Payment consistency: Ratio of min payment to max payment
    max_pay = F.greatest(*[F.col(c) for c in pay_cols])
    min_pay = F.least(*[F.col(c) for c in pay_cols])
    df = df.withColumn(
        "PAYMENT_CONSISTENCY",
        F.when(max_pay <= 0, F.lit(1.0)).otherwise(min_pay / (max_pay + 1e-5))
    )

    return df


def split_data(df: DataFrame, train_ratio: float = 0.8, seed: int = 42):
    """
    Split the dataset into training and testing sets (80/20) using seed=42.
    Evaluates and validates class preservation.
    """
    train_df, test_df = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    # Cache to prevent recalculation across multiple model runs
    train_df = train_df.cache()
    test_df = test_df.cache()

    train_total = train_df.count()
    test_total = test_df.count()
    train_default = train_df.filter(F.col("label") == 1.0).count()
    test_default = test_df.filter(F.col("label") == 1.0).count()

    print(f"Train Records: {train_total:,} (Default: {train_default:,} = {train_default/train_total*100:.2f}%)")
    print(f"Test Records:  {test_total:,} (Default: {test_default:,} = {test_default/test_total*100:.2f}%)")

    return train_df, test_df


def add_class_weights(train_df: DataFrame) -> DataFrame:
    """
    Compute class weights on training set to handle class imbalance.
    Weight = Total / (2 * Count_Class)
    """
    total = train_df.count()
    count_0 = train_df.filter(F.col("label") == 0.0).count()
    count_1 = train_df.filter(F.col("label") == 1.0).count()

    w0 = total / (2.0 * count_0)
    w1 = total / (2.0 * count_1)

    train_weighted = train_df.withColumn(
        "class_weight",
        F.when(F.col("label") == 1.0, F.lit(w1)).otherwise(F.lit(w0))
    )
    return train_weighted


def build_pipeline_stages(include_feature_engineering: bool = True):
    """
    Build Spark ML Pipeline stages for categorical encoding, assembly, and scaling.
    Returns:
    - stages: list of PipelineStage
    - feature_cols: list of input feature names assembled
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
        "TOTAL_BILL_AMT",
        "AVG_BILL_AMT",
        "TOTAL_PAY_AMT",
        "AVG_PAY_AMT",
        "MAX_BILL_AMT",
        "MIN_BILL_AMT",
        "PAY_TO_BILL_RATIO",
        "DELAYED_MONTHS_COUNT",
        "MAX_REPAYMENT_DELAY",
        "AVG_REPAYMENT_STATUS",
        "CREDIT_UTILIZATION_RATIO",
        "PAYMENT_CONSISTENCY",
    ]

    encoded_cat_cols = [f"{col}_vec" for col in categorical_cols]

    if include_feature_engineering:
        assembled_inputs = encoded_cat_cols + base_num_cols + engineered_cols
        all_feature_names = categorical_cols + base_num_cols + engineered_cols
    else:
        assembled_inputs = encoded_cat_cols + base_num_cols
        all_feature_names = categorical_cols + base_num_cols

    assembler = VectorAssembler(
        inputCols=assembled_inputs,
        outputCol="features",
        handleInvalid="skip"
    )

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=False  # sparse vector safe
    )

    stages = indexers + encoders + [assembler, scaler]
    return stages, all_feature_names, assembled_inputs


def export_preprocessing_summary():
    """
    Document preprocessing decisions and export summary CSV.
    """
    summary_data = [
        {"Step": "1. ID Column", "Action": "Dropped", "Rationale": "Arbitrary identifier without predictive signal."},
        {"Step": "2. EDUCATION Encoding", "Action": "Consolidated 0, 4, 5, 6 into 4 ('Others')", "Rationale": "Per official UCI documentation, undocumented codes grouped into category 4."},
        {"Step": "3. MARRIAGE Encoding", "Action": "Consolidated 0, 3 into 3 ('Others')", "Rationale": "Code 0 represents unassigned values; grouped with others."},
        {"Step": "4. Categorical Encoding", "Action": "StringIndexer + OneHotEncoder", "Rationale": "Ensures categorical columns are transformed into MLlib-compatible vectors."},
        {"Step": "5. Repayment History", "Action": "Preserved as ordered numeric integers (-2 to 8)", "Rationale": "Reflects monotonically increasing repayment delay severity."},
        {"Step": "6. Feature Engineering", "Action": "Generated 11 financial metrics", "Rationale": "Captures credit utilization, payment consistency, bill trends, and total delay history."},
        {"Step": "7. Feature Assembly", "Action": "VectorAssembler", "Rationale": "Combines one-hot vectors and numeric features into unified feature vector."},
        {"Step": "8. Feature Scaling", "Action": "StandardScaler (withStd=True, withMean=False)", "Rationale": "Standardizes feature scales for distance-based and linear models while preserving sparsity."},
        {"Step": "9. Train-Test Split", "Action": "80% Train, 20% Test (seed=42)", "Rationale": "Maintains target class distribution (~22.1% default) without data leakage."},
        {"Step": "10. Imbalance Handling", "Action": "Class weighting (w0=0.64, w1=2.26) & Threshold tuning", "Rationale": "Counteracts ~3.5:1 imbalance and penalizes false negatives in credit default."}
    ]
    df = pd.DataFrame(summary_data)
    out_path = ANALYSIS_DIR / "preprocessing_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved Preprocessing Summary: {out_path}")

    feat_summary = [
        {"Feature": "TOTAL_BILL_AMT", "Formula": "sum(BILL_AMT1..6)", "Description": "Cumulative 6-month billed credit amount."},
        {"Feature": "AVG_BILL_AMT", "Formula": "mean(BILL_AMT1..6)", "Description": "Average monthly expenditure statement."},
        {"Feature": "TOTAL_PAY_AMT", "Formula": "sum(PAY_AMT1..6)", "Description": "Cumulative 6-month repayments made."},
        {"Feature": "AVG_PAY_AMT", "Formula": "mean(PAY_AMT1..6)", "Description": "Average monthly repayment amount."},
        {"Feature": "MAX_BILL_AMT", "Formula": "max(BILL_AMT1..6)", "Description": "Peak billing statement across observation window."},
        {"Feature": "MIN_BILL_AMT", "Formula": "min(BILL_AMT1..6)", "Description": "Lowest billing statement across observation window."},
        {"Feature": "PAY_TO_BILL_RATIO", "Formula": "TOTAL_PAY / (TOTAL_BILL + eps)", "Description": "Proportion of total obligations satisfied by payments (clipped 0-5)."},
        {"Feature": "DELAYED_MONTHS_COUNT", "Formula": "sum(PAY_i > 0 for i in 1..6)", "Description": "Total count of distinct months with repayment delinquency."},
        {"Feature": "MAX_REPAYMENT_DELAY", "Formula": "max(PAY_0..6)", "Description": "Longest delinquent payment duration observed in months."},
        {"Feature": "AVG_REPAYMENT_STATUS", "Formula": "mean(PAY_0..6)", "Description": "Overall trend of repayment behaviour across 6 months."},
        {"Feature": "CREDIT_UTILIZATION_RATIO", "Formula": "BILL_AMT1 / LIMIT_BAL", "Description": "Current proportion of credit card limit being utilized."},
        {"Feature": "PAYMENT_CONSISTENCY", "Formula": "min(PAY_AMT) / (max(PAY_AMT) + eps)", "Description": "Ratio indicating regularity of monthly repayment amounts."}
    ]
    df_feat = pd.DataFrame(feat_summary)
    feat_path = ANALYSIS_DIR / "feature_engineering_summary.csv"
    df_feat.to_csv(feat_path, index=False)
    print(f"Saved Feature Engineering Summary: {feat_path}")


if __name__ == "__main__":
    spark = get_spark_session("PreprocessingTest")
    try:
        raw_df = load_raw_data(spark)
        clean_df = clean_and_standardize_data(raw_df)
        fe_df = engineer_financial_features(clean_df)
        train_df, test_df = split_data(fe_df)
        export_preprocessing_summary()
        print("Preprocessing test completed successfully.")
    finally:
        stop_spark_session(spark)

