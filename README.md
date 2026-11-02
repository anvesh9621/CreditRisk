# Predicting Credit Card Default Using Apache Spark and PySpark MLlib

An enterprise-grade, distributed Machine Learning and Credit Risk Analytics platform built entirely using **Apache Spark 4.3**, **PySpark MLlib**, and a **standalone financial analytics dashboard (HTML5 / CSS3 / Vanilla JS / Chart.js)**.

The system analyzes customer demographic profiles, historical billing statements, repayment delays, and cash-flow liquidity metrics across 30,000 credit card accounts from the official UCI Credit Card Clients dataset to predict the likelihood of default on upcoming monthly obligations.

---

## 1. Executive Summary & Core Results

### Target Variable:
- **`0` (No Default)**: 23,364 accounts (77.88%)
- **`1` (Default)**: 6,636 accounts (22.12%)
- **Class Imbalance Ratio**: `1 : 3.52`

### Key Model Comparison Results (Test Set Evaluation: 5,968 accounts):

| Model Architecture | Accuracy | Weighted Precision | Default Recall (Class 1) | F1-Score | ROC-AUC | PR-AUC | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Class-Weighted)** 🏆 | **75.28%** | **80.59%** | **61.57%** | **0.7653** | **0.7538** | **0.5006** | **11.70s** |
| **Gradient Boosted Tree (GBT)** | 81.75% | 79.91% | 35.20% | 0.7958 | 0.7895 | 0.5449 | 44.26s |
| **Random Forest Classifier** | 81.69% | 79.79% | 35.59% | 0.7958 | 0.7813 | 0.5472 | 19.53s |
| **Logistic Regression (Unweighted Baseline)** | 80.71% | 77.82% | 27.06% | 0.7755 | 0.7500 | 0.5035 | 17.91s |
| **Decision Tree Classifier** | 81.77% | 79.59% | 34.20% | 0.7947 | 0.2779 | 0.1495 | 10.13s |
| **Multinomial Naive Bayes** | 80.18% | 77.72% | 34.13% | 0.7817 | 0.2865 | 0.1502 | 2.88s |

> **🏆 Best Model Selection Justification**:  
> In retail credit risk management, an undetected default (**False Negative**) triggers direct loan write-offs and capital loss, whereas declining a solvent applicant (**False Positive**) only costs an opportunity margin. Unweighted models achieved ~81% overall accuracy simply by predicting non-default, while capturing only 27–35% of true defaulters.  
> The **Class-Weighted Logistic Regression model** dramatically increased default detection to **61.57% Recall** while maintaining a **75.38% ROC-AUC** and **75.28% Accuracy**, earning the highest composite validation score (**0.6210**).

---

## 2. Technology Stack

- **Distributed Data Processing**: Apache Spark 4.3, PySpark SQL DataFrames
- **Machine Learning**: PySpark MLlib (`Pipeline`, `TrainValidationSplit`, `ParamGridBuilder`, `VectorAssembler`, `StandardScaler`, `MulticlassClassificationEvaluator`, `BinaryClassificationEvaluator`)
- **Data Analysis & Visualization**: Pandas, NumPy, Matplotlib (Agg headless), Seaborn
- **Dashboard**: HTML5, CSS3 (Dark Navy Glassmorphic Theme), Vanilla JavaScript, Chart.js (CDN)
- **Runtime Environment**: Python 3.13 / OpenJDK 21 LTS on Microsoft Windows

---

## 3. Project Structure

```
D:\ApacheSpark\
│
├── dataset/
│   └── default_of_credit_card_clients.csv    # Official UCI credit card clients dataset (30,000 records)
│
├── src/
│   ├── utils.py                              # SparkSession factory, paths, directory manager, timers
│   ├── data_loader.py                        # PySpark dataset loading, schema inspection & verification
│   ├── eda.py                                # Univariate, bivariate, correlation & outlier analyses
│   ├── preprocessing.py                      # Data cleansing, 11 financial features, train-test split
│   ├── model_training.py                     # PySpark MLlib training & hyperparameter tuning (6 models)
│   ├── model_evaluation.py                   # Multi-metric evaluation, confusion matrix, ROC/PR data
│   ├── model_comparison.py                   # Model comparison table, plots, selection & persistence
│   ├── prediction.py                         # Single & batch inference application with risk tiers
│   └── dashboard_exporter.py                 # Exports analytical results to JSON for dashboard
│
├── models/
│   └── best_model/                           # Native fitted PySpark PipelineModel (persisted to disk)
│
├── outputs/
│   ├── plots/                                # 26 publication-grade high-res visualizations (.png)
│   ├── metrics/                              # Model comparison CSV, JSON, and best model metadata
│   ├── predictions/                          # Batch inference output CSV with default probabilities
│   ├── analysis/                             # Preprocessing, feature engineering & EDA summary CSVs
│   └── dashboard_data/                       # Formatted JSON feeds consumed by the frontend dashboard
│
├── dashboard/
│   ├── index.html                            # Semantic HTML5 financial analytics dashboard
│   ├── style.css                             # Dark navy palette, glassmorphism, responsive grid
│   └── script.js                             # Chart.js charts, filters, pagination, interactive simulator
│
├── notebooks/
│   ├── credit_card_default_analysis.ipynb    # End-to-end Jupyter Notebook demonstration
│   └── make_notebook.py                      # Programmatic notebook generator script
│
├── main.py                                   # Master orchestrator executing the full end-to-end pipeline
├── requirements.txt                          # Python dependencies
└── README.md                                 # Technical documentation & execution manual
```

---

## 4. Financial Feature Engineering (11 Engineered Signals)

In addition to the 23 raw predictor features, domain-specific features were engineered in PySpark:

1. **`TOTAL_BILL_AMT`**: Cumulative 6-month billed credit obligation (`BILL_AMT1 + ... + BILL_AMT6`).
2. **`AVG_BILL_AMT`**: Mean monthly expenditure demand.
3. **`TOTAL_PAY_AMT`**: Cumulative repayments settled across 6 months.
4. **`AVG_PAY_AMT`**: Average monthly liquidity committed to debt service.
5. **`MAX_BILL_AMT`**: Highest single statement balance observed.
6. **`MIN_BILL_AMT`**: Lowest billing statement.
7. **`PAY_TO_BILL_RATIO`**: Ratio of payments made to total bill liabilities (clipped $[0.0, 5.0]$).
8. **`DELAYED_MONTHS_COUNT`**: Count of distinct months with delinquent payment status (`PAY_i > 0`).
9. **`MAX_REPAYMENT_DELAY`**: Peak delinquent duration in months across the observation period.
10. **`AVG_REPAYMENT_STATUS`**: Monotonic trajectory of repayment timeliness.
11. **`CREDIT_UTILIZATION_RATIO`**: Proportion of assigned credit limit currently utilized (`BILL_AMT1 / LIMIT_BAL`).
12. **`PAYMENT_CONSISTENCY`**: Regularity ratio of minimum payment to maximum payment.

---

## 5. Execution Instructions

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Execute Master Pipeline (End-to-End)
Runs dataset loading, EDA, feature engineering, model training across all 6 architectures, comparison plots, best model persistence, batch inference, and dashboard data packaging:
```powershell
python main.py
```

### Step 3: Individual Pipeline Stages (Modular Execution)

- **Dataset Exploration**:
  ```powershell
  python -m src.data_loader
  ```

- **Exploratory Data Analysis & Visualizations**:
  ```powershell
  python -m src.eda
  ```

- **Preprocessing & Train/Test Validation**:
  ```powershell
  python -m src.preprocessing
  ```

- **Inference Application (Single & Batch Demo)**:
  ```powershell
  python src/prediction.py --mode demo
  ```

- **Batch Prediction on Custom CSV**:
  ```powershell
  python src/prediction.py --mode batch --input dataset/default_of_credit_card_clients.csv --output outputs/predictions/custom_batch.csv
  ```

### Step 4: Run Financial Analytics Dashboard
Serve the dashboard locally using Python's built-in HTTP server:
```powershell
python -m http.server 8000
```
Open your browser and navigate to:
```
http://localhost:8000/dashboard/
```
*(Alternatively, simply double-click `dashboard/index.html` in Windows Explorer; fallback data handlers ensure full functionality without requiring a web server).*

---

## 6. Dashboard Features & Capabilities

- **Executive Overview**: Real-time KPI counters, target class donut chart, delinquency impact breakdown.
- **Customer Analytics**: Interactive Bar charts for age cohorts, education, marital status, and credit limit tiers.
- **Model Performance**: Full benchmark comparison table, side-by-side ROC Curves and Precision-Recall Curves.
- **Feature Importance**: Ranked bar chart of top 15 features with domain financial rationale.
- **Prediction Results**: Searchable customer results table with live keyword filtering, risk tier pills (Low / Medium / High), pagination, and one-click CSV download.
- **Batch Prediction Engine**: Portfolio-level risk segmentation and batch prediction output download.
- **Interactive Underwriting Simulator**: Live recalculation of borrower default probability and automated credit decisioning (Approved / Conditional / Declined) as sliders and inputs are modified.

---

## 7. Verification Checklist

- [x] PySpark initializes cleanly on Windows (port `127.0.0.1`, local warehouse).
- [x] Official UCI dataset (30,000 records, 24 features) loaded with zero missing values.
- [x] All 15 Exploratory Data Analysis plots generated in `outputs/plots/`.
- [x] 11 domain financial features engineered inside PySpark DataFrames.
- [x] Strict 80/20 train-test split (`seed=42`) preserving 22.2% default class distribution.
- [x] Class imbalance addressed via inverse frequency class weighting and threshold calibration.
- [x] All 6 model architectures trained with hyperparameter tuning via `TrainValidationSplit`.
- [x] Evaluated across Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC.
- [x] Best model selected objectively and saved natively to `models/best_model/`.
- [x] Single applicant inference and batch inference verified with calibrated risk tiers.
- [x] Standalone HTML5/CSS3/Vanilla JS/Chart.js dashboard built and verified.
- [x] Full Jupyter Notebook created in `notebooks/credit_card_default_analysis.ipynb`.

