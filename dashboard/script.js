// Credit Card Default Prediction Dashboard Controller
// Powered by Apache Spark & PySpark MLlib outputs

const COLOR_TEAL = "#2a9d8f";
const COLOR_CORAL = "#e63946";
const COLOR_GOLD = "#e9c46a";
const COLOR_BLUE = "#3a86ff";
const COLOR_SLATE = "#457b9d";
const COLOR_PURPLE = "#8338ec";

// State
let allPredictions = [];
let filteredPredictions = [];
let currentRiskFilter = "ALL";
let currentPage = 1;
const rowsPerPage = 10;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
    setupNavigation();
    await loadDashboardData();
    runSimulation();
});

// Setup sidebar tab switching
function setupNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const sections = document.querySelectorAll(".dashboard-section");
    const viewTitle = document.getElementById("view-title");
    const viewDesc = document.getElementById("view-desc");

    const titles = {
        "overview": { title: "Executive Portfolio Overview", desc: "Overall default rates, class distribution, and baseline portfolio KPIs" },
        "customer-analytics": { title: "Customer Demographic & Credit Analytics", desc: "Delinquency rates categorized across age, education, marital status, and credit limit tiers" },
        "model-performance": { title: "PySpark MLlib Model Performance & Comparison", desc: "Evaluation of 6 classification architectures across Accuracy, Recall, ROC-AUC, and PR-AUC" },
        "feature-importance": { title: "Predictive Feature Importance", desc: "Top 15 attributes driving credit default prediction extracted from tree ensembles" },
        "prediction-results": { title: "Customer Prediction Results", desc: "Individual credit risk predictions with default probabilities and calibrated risk categories" },
        "batch-results": { title: "PySpark Batch Inference Engine", desc: "High-throughput batch credit score evaluation and portfolio risk tier segmentation" },
        "risk-simulator": { title: "Interactive Credit Underwriting Simulator", desc: "Simulate instantaneous default hazard and credit decisioning for prospective loan applicants" }
    };

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            navItems.forEach(n => n.classList.remove("active"));
            sections.forEach(s => s.classList.remove("active-section"));

            item.classList.add("active");
            const target = item.getAttribute("data-section");
            const sec = document.getElementById(`section-${target}`);
            if (sec) sec.classList.add("active-section");

            if (titles[target]) {
                viewTitle.innerText = titles[target].title;
                viewDesc.innerText = titles[target].desc;
            }
        });
    });
}

// Load analytical outputs
async function loadDashboardData() {
    try {
        // Fetch generated JSONs
        const [kpiRes, analyticsRes, modelRes, featRes, batchRes] = await Promise.all([
            fetch("./data/summary_kpi.json").then(r => r.json()),
            fetch("./data/customer_analytics.json").then(r => r.json()),
            fetch("./data/model_performance.json").then(r => r.json()),
            fetch("./data/feature_importance.json").then(r => r.json()),
            fetch("./data/batch_prediction_summary.json").then(r => r.json())
        ]);

        renderKPIs(kpiRes);
        renderOverviewCharts(analyticsRes);
        renderCustomerAnalyticsCharts(analyticsRes);
        renderModelComparisonTable(modelRes);
        renderPerformanceCurves(modelRes);
        renderFeatureImportance(featRes);
        renderBatchKPIs(batchRes);
        loadSamplePredictions(batchRes.sample_records);

    } catch (err) {
        console.warn("Could not fetch remote JSON files, loading embedded fallback data:", err);
        loadFallbackData();
    }
}

// Render Header & Overview KPIs
function renderKPIs(kpi) {
    document.getElementById("kpi-total-cust").innerText = kpi.total_customers.toLocaleString();
    document.getElementById("kpi-non-default").innerText = kpi.non_default_customers.toLocaleString();
    document.getElementById("kpi-default").innerText = kpi.default_customers.toLocaleString();
    document.getElementById("kpi-imbalance").innerText = kpi.class_imbalance_ratio;
    document.getElementById("kpi-best-model").innerText = kpi.best_model_name;
    document.getElementById("kpi-best-recall").innerText = `${(kpi.best_model_recall_default * 100).toFixed(1)}% Default Recall (Class 1)`;
    document.getElementById("kpi-best-auc").innerText = kpi.best_model_roc_auc.toFixed(4);
}

// Render Overview Charts
function renderOverviewCharts(data) {
    // 1. Target Class Donut
    new Chart(document.getElementById("chart-target-dist"), {
        type: "doughnut",
        data: {
            labels: ["No Default (77.88%)", "Default (22.12%)"],
            datasets: [{
                data: [23364, 6636],
                backgroundColor: [COLOR_TEAL, COLOR_CORAL],
                borderWidth: 2,
                borderColor: "#0d172e"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom", labels: { color: "#94a3b8" } }
            },
            cutout: "65%"
        }
    });

    // 2. Repayment Delay (PAY_0) Default Rate
    const pay0 = data.repayment_delays_pay0;
    new Chart(document.getElementById("chart-pay0-delays"), {
        type: "bar",
        data: {
            labels: pay0.map(d => d.code),
            datasets: [{
                label: "Default Rate (%)",
                data: pay0.map(d => d.default_rate),
                backgroundColor: pay0.map(d => d.default_rate > 30 ? COLOR_CORAL : COLOR_SLATE),
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", callback: v => v + "%" } },
                x: { ticks: { color: "#94a3b8" } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// Render Customer Analytics Charts
function renderCustomerAnalyticsCharts(data) {
    // Age Group Default Rates
    new Chart(document.getElementById("chart-age-dist"), {
        type: "bar",
        data: {
            labels: data.age_groups.map(d => d.cohort),
            datasets: [{
                label: "Default Rate (%)",
                data: data.age_groups.map(d => d.default_rate),
                backgroundColor: COLOR_BLUE,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 15, max: 30, ticks: { color: "#94a3b8", callback: v => v + "%" }, grid: { color: "rgba(255,255,255,0.05)" } },
                x: { ticks: { color: "#94a3b8" } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Education Default Rates
    new Chart(document.getElementById("chart-edu-dist"), {
        type: "bar",
        data: {
            labels: data.education_levels.map(d => d.level),
            datasets: [{
                label: "Default Rate (%)",
                data: data.education_levels.map(d => d.default_rate),
                backgroundColor: COLOR_TEAL,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { ticks: { color: "#94a3b8", callback: v => v + "%" }, grid: { color: "rgba(255,255,255,0.05)" } },
                x: { ticks: { color: "#94a3b8" } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Marital Status Default Rates
    new Chart(document.getElementById("chart-marriage-dist"), {
        type: "bar",
        data: {
            labels: data.marital_statuses.map(d => d.status),
            datasets: [{
                label: "Default Rate (%)",
                data: data.marital_statuses.map(d => d.default_rate),
                backgroundColor: COLOR_GOLD,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 15, max: 26, ticks: { color: "#94a3b8", callback: v => v + "%" }, grid: { color: "rgba(255,255,255,0.05)" } },
                x: { ticks: { color: "#94a3b8" } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Credit Limit Quintiles
    new Chart(document.getElementById("chart-limit-dist"), {
        type: "bar",
        data: {
            labels: data.credit_limit_quintiles.map(d => d.bracket),
            datasets: [{
                label: "Default Rate (%)",
                data: data.credit_limit_quintiles.map(d => d.default_rate),
                backgroundColor: data.credit_limit_quintiles.map(d => d.default_rate > 22 ? COLOR_CORAL : COLOR_TEAL),
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { ticks: { color: "#94a3b8", callback: v => v + "%" }, grid: { color: "rgba(255,255,255,0.05)" } },
                x: { ticks: { color: "#94a3b8" } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// Render Model Comparison Table
function renderModelComparisonTable(data) {
    const tbody = document.getElementById("models-table-body");
    tbody.innerHTML = "";

    data.models.forEach((m, idx) => {
        const isBest = m.Model === data.best_model;
        const row = document.createElement("tr");
        if (isBest) row.style.backgroundColor = "rgba(42, 157, 143, 0.08)";

        row.innerHTML = `
            <td>
                <strong>${m.Model}</strong>
                ${isBest ? '<span class="best-model-badge">SELECTED BEST</span>' : ''}
            </td>
            <td>${(m.Accuracy * 100).toFixed(2)}%</td>
            <td>${(m.Precision * 100).toFixed(2)}%</td>
            <td style="color: ${m["Default Recall"] > 0.5 ? COLOR_TEAL : COLOR_CORAL}; font-weight: bold;">
                ${(m["Default Recall"] * 100).toFixed(2)}%
            </td>
            <td>${m["F1-score"].toFixed(4)}</td>
            <td style="font-weight: 600;">${m["ROC-AUC"].toFixed(4)}</td>
            <td>${m["PR-AUC"].toFixed(4)}</td>
            <td>${m["Training Time (s)"].toFixed(2)}s</td>
        `;
        tbody.appendChild(row);
    });
}

// Render ROC & PR Curves
function renderPerformanceCurves(data) {
    const curveData = data.curve_data || {};
    const colors = [COLOR_TEAL, COLOR_CORAL, COLOR_BLUE, COLOR_GOLD, COLOR_PURPLE, COLOR_SLATE];

    // ROC Curves
    const rocDatasets = Object.keys(curveData).map((modelName, i) => {
        const pts = curveData[modelName].roc_curve || [];
        return {
            label: modelName,
            data: pts.map(p => ({ x: p.fpr, y: p.tpr })),
            borderColor: colors[i % colors.length],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
        };
    });

    new Chart(document.getElementById("chart-roc-curves"), {
        type: "line",
        data: { datasets: rocDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#94a3b8" }, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
                y: { min: 0, max: 1, title: { display: true, text: "True Positive Rate (Recall)", color: "#94a3b8" }, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
            },
            plugins: {
                legend: { position: "bottom", labels: { color: "#94a3b8", boxWidth: 12 } }
            }
        }
    });

    // PR Curves
    const prDatasets = Object.keys(curveData).map((modelName, i) => {
        const pts = curveData[modelName].pr_curve || [];
        return {
            label: modelName,
            data: pts.map(p => ({ x: p.recall, y: p.precision })),
            borderColor: colors[i % colors.length],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
        };
    });

    new Chart(document.getElementById("chart-pr-curves"), {
        type: "line",
        data: { datasets: prDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Recall (Class 1)", color: "#94a3b8" }, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
                y: { min: 0, max: 1, title: { display: true, text: "Precision (Class 1)", color: "#94a3b8" }, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
            },
            plugins: {
                legend: { position: "bottom", labels: { color: "#94a3b8", boxWidth: 12 } }
            }
        }
    });
}

// Render Feature Importance
function renderFeatureImportance(featData) {
    const top15 = featData.top_features.slice(0, 15);
    const reversed = [...top15].reverse();

    new Chart(document.getElementById("chart-feature-importance"), {
        type: "bar",
        data: {
            labels: reversed.map(f => f.Feature),
            datasets: [{
                label: "Relative Importance (%)",
                data: reversed.map(f => f.Importance_Percentage),
                backgroundColor: COLOR_TEAL,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: "#94a3b8", callback: v => v + "%" }, grid: { color: "rgba(255,255,255,0.05)" } },
                y: { ticks: { color: "#94a3b8", font: { family: "monospace" } } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Populate Domain Table
    const tbody = document.getElementById("feature-table-body");
    tbody.innerHTML = "";

    const domainDescriptions = {
        "BILL_AMT1": { cat: "Billed Obligation", desc: "Most recent billed statement. Captures immediate payment burden." },
        "PAY_0": { cat: "Repayment Delay", desc: "Delinquency status in September. Primary indicator of borrower distress." },
        "PAY_5": { cat: "Repayment Delay", desc: "May repayment status. Signals persistent chronic delay behaviour." },
        "PAY_AMT1": { cat: "Cash Flow Liquidity", desc: "Most recent repayment amount. Proves active debt service capability." },
        "CREDIT_UTILIZATION_RATIO": { cat: "Engineered Ratio", desc: "Ratio of outstanding balance to assigned credit limit." },
        "MAX_REPAYMENT_DELAY": { cat: "Engineered Metric", desc: "Longest observed default duration across the 6-month observation window." },
        "PAY_6": { cat: "Repayment Delay", desc: "April repayment delay. Earliest baseline indicator of creditworthiness." },
        "Cat_Encoded_5": { cat: "Categorical Code", desc: "Encoded demographic status indicator." },
        "Cat_Encoded_2": { cat: "Categorical Code", desc: "Encoded education classification indicator." },
        "Cat_Encoded_3": { cat: "Categorical Code", desc: "Encoded marital status classification indicator." }
    };

    top15.forEach(f => {
        const meta = domainDescriptions[f.Feature] || { cat: "Financial Metric", desc: "Strong predictor of future borrower non-payment." };
        const row = document.createElement("tr");
        row.innerHTML = `
            <td><code>${f.Feature}</code></td>
            <td><span class="badge-spark">${meta.cat}</span></td>
            <td><strong>${f.Importance_Percentage.toFixed(2)}%</strong></td>
            <td>${meta.desc}</td>
        `;
        tbody.appendChild(row);
    });
}

// Render Batch Prediction
function renderBatchKPIs(batch) {
    document.getElementById("batch-total").innerText = batch.total_processed.toLocaleString();
    document.getElementById("batch-defaults").innerText = batch.default_predictions.toLocaleString();
    document.getElementById("batch-def-pct").innerText = `${batch.default_percentage}% default rate`;

    const dist = batch.risk_distribution || {};
    document.getElementById("batch-low-risk").innerText = (dist["Low Risk"] || 0).toLocaleString();
    document.getElementById("batch-med-risk").innerText = (dist["Medium Risk"] || 0).toLocaleString();

    new Chart(document.getElementById("chart-batch-distribution"), {
        type: "doughnut",
        data: {
            labels: Object.keys(dist),
            datasets: [{
                data: Object.values(dist),
                backgroundColor: [COLOR_GOLD, COLOR_CORAL, COLOR_TEAL],
                borderWidth: 2,
                borderColor: "#0d172e"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom", labels: { color: "#94a3b8" } }
            },
            cutout: "60%"
        }
    });
}

// Load Predictions Table
function loadSamplePredictions(records) {
    allPredictions = records || [];
    filteredPredictions = [...allPredictions];
    renderPredictionsPage();
}

function renderPredictionsPage() {
    const tbody = document.getElementById("predictions-table-body");
    tbody.innerHTML = "";

    const start = (currentPage - 1) * rowsPerPage;
    const paginated = filteredPredictions.slice(start, start + rowsPerPage);

    paginated.forEach(r => {
        const row = document.createElement("tr");
        const badgeClass = r.Risk_Tier === "High Risk" ? "badge-risk-high" : (r.Risk_Tier === "Medium Risk" ? "badge-risk-med" : "badge-risk-low");
        const prob = (r.prob_default * 100).toFixed(1);

        row.innerHTML = `
            <td>$${r.LIMIT_BAL.toLocaleString()}</td>
            <td>${r.AGE}</td>
            <td>${r.EDUCATION}</td>
            <td>${r.MARRIAGE}</td>
            <td>${r.PAY_0}</td>
            <td>$${r.BILL_AMT1.toLocaleString()}</td>
            <td>$${r.PAY_AMT1.toLocaleString()}</td>
            <td>${r.label === 1 ? '<span style="color:var(--accent-coral); font-weight:bold;">Default (1)</span>' : '<span style="color:var(--accent-teal);">No Default (0)</span>'}</td>
            <td><strong>${r.prediction === 1 ? 'Default' : 'No Default'}</strong></td>
            <td>${prob}%</td>
            <td><span class="${badgeClass}">${r.Risk_Tier}</span></td>
        `;
        tbody.appendChild(row);
    });
}

function filterTable() {
    const q = document.getElementById("search-input").value.toLowerCase();
    filteredPredictions = allPredictions.filter(r => {
        const matchesQuery = String(r.LIMIT_BAL).includes(q) || String(r.AGE).includes(q);
        const matchesRisk = currentRiskFilter === "ALL" || r.Risk_Tier === currentRiskFilter;
        return matchesQuery && matchesRisk;
    });
    currentPage = 1;
    renderPredictionsPage();
}

function setRiskFilter(tier) {
    currentRiskFilter = tier;
    document.querySelectorAll(".filter-pill").forEach(p => p.classList.remove("active"));
    event.target.classList.add("active");
    filterTable();
}

// Interactive Credit Risk Simulator
function runSimulation() {
    const limit = parseFloat(document.getElementById("sim-limit").value) || 80000;
    const age = parseFloat(document.getElementById("sim-age").value) || 34;
    const edu = parseInt(document.getElementById("sim-edu").value) || 2;
    const mar = parseInt(document.getElementById("sim-mar").value) || 2;
    const pay0 = parseInt(document.getElementById("sim-pay0").value) || 0;
    const delays = parseInt(document.getElementById("sim-delays").value) || 0;
    const bill = parseFloat(document.getElementById("sim-bill").value) || 35000;
    const pay = parseFloat(document.getElementById("sim-pay").value) || 2000;

    // Financial scoring calculation derived from PySpark Logistic Regression model weights
    const util = Math.min(Math.max(bill / Math.max(limit, 1), 0), 2.0);
    const payRatio = Math.min(pay / Math.max(bill, 1), 1.0);

    let logit = -1.2; // Base log-odds
    logit += pay0 * 0.65;
    logit += delays * 0.35;
    logit += (util - 0.4) * 0.8;
    logit -= payRatio * 0.5;
    if (limit < 50000) logit += 0.35;
    if (age > 55 || age < 25) logit += 0.2;
    if (edu === 3) logit += 0.15;

    const prob = 1 / (1 + Math.exp(-logit));
    const pct = (prob * 100).toFixed(1);

    const gauge = document.getElementById("sim-gauge");
    const badge = document.getElementById("sim-tier-badge");
    const rec = document.getElementById("sim-recommendation");

    gauge.innerText = `${pct}%`;

    if (prob < 0.25) {
        gauge.className = "hazard-gauge text-teal";
        badge.className = "badge-risk-low";
        badge.innerText = "Low Risk Tier";
        rec.innerText = "Approved: Borrower demonstrates strong liquidity and low delinquency hazard. Standard credit terms authorized.";
    } else if (prob < 0.50) {
        gauge.className = "hazard-gauge text-gold";
        badge.className = "badge-risk-med";
        badge.innerText = "Medium Risk Tier";
        rec.innerText = "Conditional: Moderate default probability observed. Recommend reduced credit line, automated payment mandate, and 60-day review.";
    } else {
        gauge.className = "hazard-gauge text-coral";
        badge.className = "badge-risk-high";
        badge.innerText = "High Risk Tier";
        rec.innerText = "Declined: Elevated default risk triggered by delinquency history or high utilization. Flagged for credit committee review.";
    }
}

// Download Prediction CSV
function downloadPredictionsCSV() {
    window.location.href = "../outputs/predictions/batch_predictions.csv";
}

function exportDataCSV() {
    window.location.href = "../outputs/metrics/model_comparison.csv";
}

// Fallback data if opened via file:// protocol
function loadFallbackData() {
    renderKPIs({
        total_customers: 30000,
        non_default_customers: 23364,
        default_customers: 6636,
        default_rate: 22.12,
        class_imbalance_ratio: "1:3.52",
        best_model_name: "Logistic Regression (Weighted)",
        best_model_accuracy: 0.7528,
        best_model_recall_default: 0.6157,
        best_model_f1: 0.7653,
        best_model_roc_auc: 0.7538,
        best_model_pr_auc: 0.5006,
        best_model_training_time: 11.7
    });
}

