# 📊 Financial Analytics & Corporate Valuation Analysis of Leading Indian Companies

> **IBM SkillsBuild Data Analytics Internship**
> In association with **AICTE** and **BharatCares**

---

## 📋 Table of Contents

1. [Program Details](#-program-details)
2. [Project Overview](#-project-overview)
3. [Dataset Overview](#-dataset-overview)
4. [Tech Stack & Tools](#-tech-stack--tools)
5. [Repository Structure](#-repository-structure)
6. [Installation & Setup](#-installation--setup)
7. [Key Analytical Insights](#-key-analytical-insights)
8. [Output Files](#-output-files)
9. [Acknowledgements](#-acknowledgements)

---

## 🎓 Program Details

| Field | Details |
|---|---|
| **Student** | Kukku Choudhary |
| **Institution** | Samrat Ashok Technological Institute, Vidisha |
| **Program** | IBM SkillsBuild Data Analytics Internship |
| **Organized by** | IBM SkillsBuild |
| **In association with** | AICTE & BharatCares |
| **Domain** | Data Analytics · Financial Analytics · Machine Learning |

---

## 🔍 Project Overview

This project performs an end-to-end financial data analysis on **459 publicly listed Indian companies**, covering:

- **Data Cleaning** — Handling comma-formatted numbers, percentage strings, and null values
- **Exploratory Data Analysis (EDA)** — Summary statistics across all numerical attributes
- **Data Visualization** — Market share rankings, valuation distributions, and correlation patterns
- **Machine Learning** — Linear Regression model predicting market capitalisation from quarterly sales

---

## 📂 Dataset Overview

**File:** `Financial-Analytics-data1.csv`

| Attribute | Description |
|---|---|
| `company_name` | Name of the listed Indian company |
| `market_cap_crore` | Market capitalisation in ₹ Crore |
| `sales_qtr_crore` | Quarterly sales revenue in ₹ Crore |
| `market_cap_category` | Size segment — Small Cap / Mid Cap / Large Cap |
| `sales_qtr_category` | Sales tier — Low / Medium / High Sales |
| `market_cap_to_sales` | Valuation multiple (Market Cap ÷ Quarterly Sales) |
| `market_cap_share` | Company's share of total market capitalisation (%) |

- **Records:** 459 companies
- **Scope:** Leading listed companies across sectors on Indian stock exchanges
- **Key Insight:** Highly skewed distribution — top 10 companies account for a disproportionate share of total market capitalisation

---

## 🛠 Tech Stack & Tools

| Category | Technology |
|---|---|
| **Language** | Python 3.x |
| **Data Manipulation** | Pandas 2.2.3, NumPy 1.26.4 |
| **Visualization** | Matplotlib 3.9.4, Seaborn 0.13.2 |
| **Machine Learning** | Scikit-Learn 1.6.1 |
| **AI Assistant** | IBM Bob AI Assistant |
| **Notebook Environment** | Jupyter Notebook |

---

## 🗂 Repository Structure

```
├── Financial-Analytics-data1.csv          # Raw dataset (459 records)
├── KukkuChoudhary_FinancialAnalytics.ipynb # Main analysis notebook
├── financial_analytics.py                 # EDA & visualization script
├── financial_ml.py                        # Linear regression ML script
├── requirements.txt                       # Pinned Python dependencies
├── top_10_market_share.png                # Chart 1 — Top 10 by market cap share
├── market_cap_to_sales_boxplot.png        # Chart 2 — Valuation ratio boxplot
├── correlation_matrix.png                 # Chart 3 — Correlation heatmap
├── regression_plot.png                    # Chart 4 — Regression scatter plot
└── README.md                              # Project documentation (this file)
```

---

## ⚙ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required libraries at their pinned versions:

```
pandas==2.2.3
numpy==1.26.4
matplotlib==3.9.4
seaborn==0.13.2
scikit-learn==1.6.1
```

### 3. Launch the Jupyter Notebook

```bash
jupyter notebook KukkuChoudhary_FinancialAnalytics.ipynb
```

> **Alternatively**, run the standalone Python scripts directly:
>
> ```bash
> python financial_analytics.py   # EDA + 3 visualization charts
> python financial_ml.py          # Linear regression + regression_plot.png
> ```

---

## 📈 Key Analytical Insights

### Market Concentration

- The **top 10 companies** (led by Reliance Industries at ~4.40%) collectively dominate total market capitalisation, reflecting high concentration in Indian equities.
- **Large Cap** companies exhibit significantly higher `market_cap_to_sales` multiples compared to Mid Cap and Small Cap peers, indicating premium valuations.
- The correlation heatmap reveals a **strong positive relationship** between `market_cap_crore` and `sales_qtr_crore`, confirming that revenue scale is a key driver of market value.

### Linear Regression Model Performance

| Metric | Result |
|---|---|
| **Feature (X)** | `sales_qtr_clean` — Quarterly Sales (₹ Crore) |
| **Target (y)** | `market_cap_clean` — Market Capitalisation (₹ Crore) |
| **Train / Test Split** | 80% / 20% (`random_state=42`) |
| **R² Score** | Model explains a significant proportion of variance in market cap |
| **RMSE** | Expressed in ₹ Crore; lower values confirm predictive accuracy |

> 📌 Actual R² and RMSE values are printed to the console when the script is executed, as they depend on the final cleaned record count.

### Visualizations Generated

| Chart | File | Description |
|---|---|---|
| Horizontal Bar | `top_10_market_share.png` | Top 10 companies ranked by market cap share (%) |
| Boxplot | `market_cap_to_sales_boxplot.png` | Valuation multiple distribution by cap category |
| Heatmap | `correlation_matrix.png` | Pearson correlation across all numerical features |
| Scatter + Line | `regression_plot.png` | Actual vs predicted market cap with regression line |

---

## 🙏 Acknowledgements

- **IBM SkillsBuild** for providing the internship platform and learning resources
- **AICTE** and **BharatCares** for organizing and supporting this program
- **IBM Bob AI Assistant** for assisting with code generation, analysis, and documentation

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-2.2.3-green?style=flat-square" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?style=flat-square" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/IBM%20SkillsBuild-Internship-054ADA?style=flat-square" alt="IBM SkillsBuild"/>
</p>
