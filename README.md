# Student Performance Analytics Using Hadoop and Apache Spark

An end-to-end academic intelligence platform for analyzing student performance, attendance patterns, department comparisons, and automated early-risk identification with rule-based intervention recommendations.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.5%2B-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Hadoop HDFS](https://img.shields.io/badge/Hadoop_HDFS-3.x-yellow.svg)](https://hadoop.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Project Overview

Academic institutions often face difficulties identifying at-risk students before final evaluations. **Student Performance Analytics** bridges this gap by unifying distributed data processing via **Apache Spark** and **Hadoop HDFS** with a responsive, interactive **Streamlit** dashboard.

The application ingests institutional datasets (~10,000 records across 21 academic and demographic attributes), aggregates performance benchmarks across departments, and screens students using configurable rule-based risk criteria.

---

## 🚀 2. Key Features

- **Executive KPI Dashboard**: Displays true student-weighted average score, weighted attendance, cohort count, and real-time at-risk ratios.
- **Global Department Filtering**: Dynamic multi-department filtering (`Business`, `CS`, `Engineering`, `Mathematics`) that synchronizes across all analytical views with reset capabilities.
- **Department-Level Analytics**: Comparative visualizations of total score and attendance, departmental summary table, and automated institutional insights (top scoring department, highest attendance, enrollment spread).
- **Student Performance Explorer**: Granular filtering by department and letter grade (`A`, `B`, `C`, `D`, `F`), case-insensitive student ID search, grade distribution histograms, and attendance vs. score scatter plots.
- **Rule-Based At-Risk Screening**: Transparent student screening with configurable sliders (default: attendance < 60% or score < 50) using strict inequality rules.
- **Risk Severity Classification**:
  - 🔴 **High Priority**: Both low attendance and low score.
  - 🟡 **Medium Priority**: Exactly one risk condition breached.
  - 🟢 **No Flag**: Student meets or exceeds academic thresholds.
- **Actionable Intervention Plans**: Pre-mapped, non-judgmental intervention guidance for academic advisors and department heads.
- **One-Click CSV Reports**: Export filtered data on demand:
  1. Department Summary CSV
  2. Filtered Student Records CSV
  3. At-Risk Student Screening Report CSV
  4. Student Intervention Report CSV
- **Data Quality & Spark Health Monitor**: Built-in schema validation, missing value diagnostics, duplicate checks, and PySpark/HDFS storage status.

---

## 🏗️ 3. Architecture & Data Flow

```text
[ Raw Student Dataset ] (data/raw/students.csv: 10,000 rows, 21 cols)
          │
          ├───► [ Apache Spark Job / PySpark ] (spark_jobs/department_analytics.py)
          │            │
          │            ▼
          │     [ Hadoop HDFS / Local Fallback ]
          │     (hdfs://localhost:9000/... OR data/processed/department_summary/)
          │            │
          │            ▼
          └────► [ Modular Data Engine (src/) ]
                       ├── data_loader.py   (Schema validation & HDFS/local ingestion)
                       ├── analytics.py     (Weighted KPIs & department metrics)
                       ├── risk_engine.py   (Strict rule-based risk classification)
                       └── interventions.py (Advising recommendations mapping)
                               │
                               ▼
                   [ Streamlit Web Dashboard ] (dashboard/app.py)
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
     [ KPI Overview ]   [ Dept Analytics ]  [ Risk Screening ]
            │                  │                  │
            └──────────────────┴──────────────────┘
                               ▼
                  [ Interactive CSV Downloads ]
```

---

## 📁 4. Project Folder Structure

```text
Student-Performance-Analytics/
├── .streamlit/
│   └── config.toml                  # Streamlit institutional dark theme configuration
├── dashboard/
│   └── app.py                       # Main Streamlit web application
├── data/
│   ├── raw/
│   │   └── students.csv             # Primary student performance dataset (10,000 records)
│   └── processed/
│       └── department_summary/      # Spark aggregation output directory
│           └── part-00000-*.csv     # Precomputed department summary
├── spark_jobs/
│   ├── department_analytics.py      # PySpark department aggregation job (HDFS + Local)
│   └── student_analytics.py         # PySpark student statistics job
├── src/
│   ├── __init__.py                  # Core package init
│   ├── analytics.py                 # Weighted KPIs, filters, distributions, and insights
│   ├── data_loader.py               # Dataset ingestion and validation diagnostics
│   ├── interventions.py             # Rule-based academic intervention mapping
│   └── risk_engine.py               # Strict risk screening and severity engine
├── tests/
│   └── test_analytics.py            # Comprehensive 18-suite pytest verification
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation and user manual
```

---

## ⚙️ 5. Technology Stack

- **Frontend & UI**: Streamlit, Plotly Express & Graph Objects, Custom CSS
- **Data Engineering**: Apache Spark 3.5+, PySpark, Hadoop 3.x HDFS
- **Analytics & Processing**: Python 3.10+, Pandas, NumPy
- **Testing & Quality**: Pytest
- **Platform Compatibility**: Windows 10/11, Ubuntu 22.04+ (WSL2), Streamlit Community Cloud

---

## 💻 6. Local Setup and Installation

### A. Clone the Repository
```bash
git clone https://github.com/Aayushjha40/Student-Performance-Analytics.git
cd Student-Performance-Analytics
```

### B. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Ubuntu / WSL2 (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### C. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 7. Running the Application

Launch the Streamlit web dashboard locally:

```bash
streamlit run dashboard/app.py
```

The application will automatically start and open in your default browser at:
`http://localhost:8501`

---

## ⚡ 8. Running Apache Spark Jobs

The PySpark jobs support both **Hadoop HDFS** and **Local Filesystem** execution with automatic fallback.

### A. Running on Hadoop HDFS (Ubuntu WSL2)
Ensure your Hadoop daemons (NameNode, DataNode) are running:
```bash
# Start HDFS daemons
start-dfs.sh

# Verify HDFS health
hdfs dfsadmin -report

# Upload raw student dataset to HDFS
hdfs dfs -mkdir -p /student-performance/raw/
hdfs dfs -put -f data/raw/students.csv /student-performance/raw/

# Submit Spark job
spark-submit spark_jobs/department_analytics.py
```

### B. Running Locally (Without Hadoop Cluster)
If Hadoop/HDFS is not running, execute the Spark script directly with the `--local` flag:
```bash
python spark_jobs/department_analytics.py --local
```
Or for student analytics:
```bash
python spark_jobs/student_analytics.py --local
```

The job aggregates metrics across all departments and outputs the processed summary directly into `data/processed/department_summary/`.

---

## 🧪 9. Running Automated Tests

Run the full pytest suite to verify all 18 requirements (data loading, schema validation, weighted averages, filters, risk rules, and CSV exports):

```bash
pytest tests/ -v
```

All test cases utilize deterministic synthetic fixtures to validate edge cases (including strict `<` threshold boundaries at 60% and 50 points).

---

## 🎯 10. Risk Rules and Screening Logic

The early-intervention system classifies students strictly by:
$$\text{Flagged} \iff (\text{Attendance} < \text{Threshold}_{\text{att}}) \lor (\text{Total\_Score} < \text{Threshold}_{\text{score}})$$

| Condition | Risk Reason | Severity Priority | Recommended Action Pathway |
| :--- | :--- | :--- | :--- |
| $\text{Att} < T_{\text{att}}$ AND $\text{Score} < T_{\text{score}}$ | `Low attendance + Low score` | 🔴 **High priority** | Comprehensive advising & faculty mentor assignment |
| $\text{Att} < T_{\text{att}}$ AND $\text{Score} \ge T_{\text{score}}$ | `Low attendance` | 🟡 **Medium priority** | Attendance tracking & barrier identification |
| $\text{Att} \ge T_{\text{att}}$ AND $\text{Score} < T_{\text{score}}$ | `Low score` | 🟡 **Medium priority** | Subject-matter tutoring & academic skills workshops |
| $\text{Att} \ge T_{\text{att}}$ AND $\text{Score} \ge T_{\text{score}}$ | `No flag` | 🟢 **No flag** | Standard academic trajectory |

*Note: Equality at the threshold (e.g. attendance exactly 60.0% or score exactly 50.0) is treated as meeting expectations and is not flagged.*

---

## 🌐 11. Streamlit Community Cloud Deployment

To deploy this project to Streamlit Community Cloud:

1. Push your latest commits to your GitHub repository:
   ```bash
   git add .
   git commit -m "feat: complete student performance analytics dashboard"
   git push origin main
   ```
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**, select the repository `Aayushjha40/Student-Performance-Analytics`, and branch `main`.
4. Set the **Main file path** to: `dashboard/app.py`.
5. Click **"Deploy"**. The app will automatically install dependencies from `requirements.txt` and load the committed processed results.

---

## 📝 12. Known Assumptions & Limitations

- **Dataset Schema**: The dataset contains 10,000 synthetic student records with valid ranges for attendance (50%–100%) and scores (50–100). When lowering thresholds below 50, score breaches reflect real-time user selections.
- **HDFS Availability**: In cloud environments (e.g. Streamlit Cloud), local HDFS is unavailable; the dashboard automatically utilizes precomputed Spark summary files or dynamic fallback without crashing.
- **Screening Scope**: Flags and recommendations are rule-based screening indicators intended for educator decision support, not definitive judgments of student potential.
