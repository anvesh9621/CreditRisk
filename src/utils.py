import os
import sys
import json
import time
from pathlib import Path
from contextlib import contextmanager

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "default_of_credit_card_clients.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
DASHBOARD_DATA_DIR = OUTPUTS_DIR / "dashboard_data"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model"

# Configure Hadoop and Python environments for Spark on Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if os.path.exists("D:\\hadoop"):
    os.environ["HADOOP_HOME"] = "D:\\hadoop"
    if "D:\\hadoop\\bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = "D:\\hadoop\\bin;" + os.environ.get("PATH", "")


def ensure_directories():
    """Ensure all required project output directories exist."""
    for path in [
        PLOTS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        ANALYSIS_DIR,
        DASHBOARD_DATA_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_spark_session(app_name: str = "CreditCardDefaultPrediction"):
    """
    Initialize and return an optimized SparkSession for local execution on Windows.
    """
    from pyspark.sql import SparkSession

    warehouse_dir = str(BASE_DIR / "spark-warehouse")

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def stop_spark_session(spark):
    """Safely stop active SparkSession."""
    if spark:
        try:
            spark.stop()
        except Exception:
            pass


@contextmanager
def timer(step_name: str):
    """Context manager to measure and report execution time."""
    start_time = time.time()
    print(f"\n[START] {step_name}...")
    yield
    elapsed = time.time() - start_time
    print(f"[DONE]  {step_name} completed in {elapsed:.2f} seconds.")


def save_json(data: dict, filepath: Path):
    """Serialize dictionary to formatted JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved JSON: {filepath}")


def load_json(filepath: Path) -> dict:
    """Load JSON file into Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

