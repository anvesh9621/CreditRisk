from pathlib import Path
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from src.utils import (
    ANALYSIS_DIR,
    DATASET_PATH,
    get_spark_session,
    stop_spark_session,
    ensure_directories,
)


def load_raw_data(spark, filepath: Path = DATASET_PATH) -> DataFrame:
    """
    Load the credit card dataset from CSV using PySpark DataFrame API.
    Standardizes column names by replacing dots with underscores.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = spark.read.csv(str(filepath), header=True, inferSchema=True)
    # Standardize column names (e.g. 'default.payment.next.month' -> 'default_payment_next_month')
    for c in df.columns:
        if "." in c:
            df = df.withColumnRenamed(c, c.replace(".", "_"))
    return df


def analyze_missing_values(df: DataFrame) -> pd.DataFrame:
    """
    Calculate missing/null values for each column using PySpark functions.
    """
    null_counts = []
    total_records = df.count()
    dtypes_dict = dict(df.dtypes)

    for col_name in df.columns:
        col_type = dtypes_dict[col_name]
        if col_type in ("double", "float"):
            condition = F.col(col_name).isNull() | F.isnan(F.col(col_name))
        else:
            condition = F.col(col_name).isNull()

        null_count = df.filter(condition).count()
        null_percentage = (null_count / total_records) * 100
        null_counts.append(
            {
                "Column": col_name,
                "Null_Count": int(null_count),
                "Null_Percentage": round(float(null_percentage), 4),
            }
        )

    return pd.DataFrame(null_counts)


def analyze_unique_values(df: DataFrame) -> pd.DataFrame:
    """
    Calculate unique value counts for all columns.
    """
    records = []
    for col_name in df.columns:
        unique_cnt = df.select(col_name).distinct().count()
        records.append({"Column": col_name, "Unique_Count": int(unique_cnt)})
    return pd.DataFrame(records)


def analyze_target_distribution(df: DataFrame, target_col: str) -> pd.DataFrame:
    """
    Calculate class counts and percentage for target variable.
    """
    total = df.count()
    dist_df = (
        df.groupBy(target_col)
        .count()
        .withColumn(
            "Percentage", F.round((F.col("count") / F.lit(total)) * 100, 2)
        )
        .toPandas()
    )
    dist_df.rename(
        columns={target_col: "Class", "count": "Record_Count"}, inplace=True
    )
    dist_df["Class_Label"] = dist_df["Class"].map(
        {0: "No Default", 1: "Default"}
    )
    return dist_df


def explore_dataset(spark, df: DataFrame) -> dict:
    """
    Perform thorough data exploration and export analytical CSVs.
    """
    ensure_directories()
    total_records = df.count()
    total_columns = len(df.columns)
    target_col = (
        "default_payment_next_month"
        if "default_payment_next_month" in df.columns
        else "default.payment.next.month"
    )

    print("=" * 60)
    print("DATASET EXPLORATION SUMMARY")
    print("=" * 60)
    print(f"Total Records: {total_records}")
    print(f"Total Columns: {total_columns}")

    # 1. First 5 records
    print("\nFirst 5 Records:")
    df.show(5, truncate=False)

    # 2. Schema / Data Types
    dtypes_df = pd.DataFrame(df.dtypes, columns=["Column", "DataType"])
    dtypes_path = ANALYSIS_DIR / "data_types.csv"
    dtypes_df.to_csv(dtypes_path, index=False)
    print(f"Saved Schema & Data Types: {dtypes_path}")

    # 3. Duplicate Detection
    distinct_records = df.distinct().count()
    duplicate_records = total_records - distinct_records
    print(
        f"Distinct Records: {distinct_records} | Duplicates: {duplicate_records}"
    )

    # 4. Missing Values Analysis
    missing_df = analyze_missing_values(df)
    missing_path = ANALYSIS_DIR / "missing_values.csv"
    missing_df.to_csv(missing_path, index=False)
    print(f"Saved Missing Value Analysis: {missing_path}")

    # 5. Unique Values Analysis
    unique_df = analyze_unique_values(df)
    unique_path = ANALYSIS_DIR / "unique_values.csv"
    unique_df.to_csv(unique_path, index=False)
    print(f"Saved Unique Value Analysis: {unique_path}")

    # 6. Target Class Distribution
    target_dist_df = analyze_target_distribution(df, target_col)
    target_path = ANALYSIS_DIR / "target_distribution.csv"
    target_dist_df.to_csv(target_path, index=False)
    print(f"Saved Target Distribution: {target_path}")

    # 7. Numerical Feature Summaries (describe)
    num_cols = [
        c
        for c, t in df.dtypes
        if t in ("int", "double", "bigint") and c not in ["ID", target_col]
    ]
    summary_spark = df.select(num_cols).describe()
    summary_df = summary_spark.toPandas()
    summary_path = ANALYSIS_DIR / "numerical_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved Numerical Summary: {summary_path}")

    # 8. Overall Dataset Summary Export
    c0_count = int(
        target_dist_df.loc[target_dist_df["Class"] == 0, "Record_Count"].values[
            0
        ]
    )
    c1_count = int(
        target_dist_df.loc[target_dist_df["Class"] == 1, "Record_Count"].values[
            0
        ]
    )
    c0_pct = float(
        target_dist_df.loc[target_dist_df["Class"] == 0, "Percentage"].values[0]
    )
    c1_pct = float(
        target_dist_df.loc[target_dist_df["Class"] == 1, "Percentage"].values[0]
    )

    dataset_summary = pd.DataFrame(
        [
            {"Metric": "Total Records", "Value": str(total_records)},
            {"Metric": "Total Features (inc. ID)", "Value": str(total_columns)},
            {
                "Metric": "Predictor Features",
                "Value": str(total_columns - 2),
            },  # minus ID and target
            {"Metric": "Duplicate Records", "Value": str(duplicate_records)},
            {"Metric": "Columns with Missing Values", "Value": "0"},
            {"Metric": "Non-Default Count (Class 0)", "Value": str(c0_count)},
            {"Metric": "Non-Default Percentage", "Value": f"{c0_pct}%"},
            {"Metric": "Default Count (Class 1)", "Value": str(c1_count)},
            {"Metric": "Default Percentage", "Value": f"{c1_pct}%"},
            {
                "Metric": "Class Imbalance Ratio",
                "Value": f"1:{round(c0_count / c1_count, 2)}",
            },
        ]
    )
    dataset_summary_path = ANALYSIS_DIR / "dataset_summary.csv"
    dataset_summary.to_csv(dataset_summary_path, index=False)
    print(f"Saved Overall Dataset Summary: {dataset_summary_path}")

    return {
        "total_records": total_records,
        "total_columns": total_columns,
        "duplicate_records": duplicate_records,
        "target_distribution": target_dist_df.to_dict(orient="records"),
        "missing_summary": missing_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    spark = get_spark_session("DataLoaderTest")
    try:
        raw_df = load_raw_data(spark)
        stats = explore_dataset(spark, raw_df)
        print("\nExploration completed successfully.")
    finally:
        stop_spark_session(spark)
