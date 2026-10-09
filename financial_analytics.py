"""
Financial Analytics - Data Analysis & Visualization Script
Dataset: Financial-Analytics-data1.csv (459 records)
Columns: company_name, market_cap_crore, sales_qtr_crore,
         market_cap_category, sales_qtr_category,
         market_cap_to_sales, market_cap_share
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN THE DATASET
# ─────────────────────────────────────────────

# Load raw CSV
df = pd.read_csv("Financial-Analytics-data1.csv")

# ── Clean numerical columns ──────────────────
# market_cap_crore and sales_qtr_crore contain comma-formatted numbers
# (e.g. "583,436.72") — strip commas before converting to float.
df["market_cap_crore"] = (
    df["market_cap_crore"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace("nan", np.nan)
    .astype(float)
)

df["sales_qtr_crore"] = (
    df["sales_qtr_crore"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace("nan", np.nan)
    .astype(float)
)

# market_cap_to_sales is already numeric but may contain stray whitespace
df["market_cap_to_sales"] = pd.to_numeric(
    df["market_cap_to_sales"].astype(str).str.strip(), errors="coerce"
)

# market_cap_share contains a '%' suffix (e.g. "4.40%") — strip it and convert
df["market_cap_share"] = (
    df["market_cap_share"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
    .replace("nan", np.nan)
    .astype(float)
)

# Drop rows where ALL numerical columns are null (fully empty records)
numerical_cols = ["market_cap_crore", "sales_qtr_crore",
                  "market_cap_to_sales", "market_cap_share"]
df.dropna(subset=numerical_cols, how="all", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Missing values per column:\n{df[numerical_cols].isnull().sum()}\n")


# ─────────────────────────────────────────────
# 2. SUMMARY STATISTICS
# ─────────────────────────────────────────────

print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

for col in numerical_cols:
    series = df[col].dropna()
    print(f"\n── {col} ──")
    print(f"  Mean   : {series.mean():>15,.2f}")
    print(f"  Median : {series.median():>15,.2f}")
    print(f"  Min    : {series.min():>15,.2f}")
    print(f"  Max    : {series.max():>15,.2f}")
    print(f"  Std Dev: {series.std():>15,.2f}")

# Also display the full pandas describe() table for reference
print("\n── Full describe() ──")
print(df[numerical_cols].describe().round(2).to_string())
print("=" * 60)


# ─────────────────────────────────────────────
# 3a. CHART 1 — Top 10 Companies by market_cap_share
#     (Horizontal Bar Plot)
# ─────────────────────────────────────────────

# Sort descending and take the top 10 rows
top10 = (
    df[["company_name", "market_cap_share"]]
    .dropna()
    .sort_values("market_cap_share", ascending=False)
    .head(10)
    # Reverse so the largest bar appears at the top of the chart
    .iloc[::-1]
)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    top10["company_name"],
    top10["market_cap_share"],
    color=sns.color_palette("Blues_d", len(top10)),
    edgecolor="white",
)

# Annotate each bar with its exact value
for bar, val in zip(bars, top10["market_cap_share"]):
    ax.text(
        bar.get_width() + 0.04,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}%",
        va="center",
        fontsize=9,
        color="#333333",
    )

ax.set_xlabel("Market Cap Share (%)", fontsize=11)
ax.set_title("Top 10 Companies by Market Cap Share", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("top_10_market_share.png", dpi=150)
plt.close()
print("Saved → top_10_market_share.png")


# ─────────────────────────────────────────────
# 3b. CHART 2 — Boxplot: market_cap_to_sales across market_cap_category
# ─────────────────────────────────────────────

# Remove extreme outliers (beyond 3 std-devs) so the chart remains readable
mts = df[["market_cap_category", "market_cap_to_sales"]].dropna()
mu, sigma = mts["market_cap_to_sales"].mean(), mts["market_cap_to_sales"].std()
mts_clean = mts[np.abs(mts["market_cap_to_sales"] - mu) <= 3 * sigma]

# Define a natural ordering for the categories
cat_order = ["Small Cap", "Mid Cap", "Large Cap"]
cat_order = [c for c in cat_order if c in mts_clean["market_cap_category"].unique()]

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(
    data=mts_clean,
    x="market_cap_category",
    y="market_cap_to_sales",
    order=cat_order,
    palette="Set2",
    linewidth=1.2,
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
    ax=ax,
)

ax.set_xlabel("Market Cap Category", fontsize=11)
ax.set_ylabel("Market Cap to Sales Ratio", fontsize=11)
ax.set_title(
    "Market Cap-to-Sales Ratio Distribution\nby Market Cap Category",
    fontsize=13,
    fontweight="bold",
)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("market_cap_to_sales_boxplot.png", dpi=150)
plt.close()
print("Saved → market_cap_to_sales_boxplot.png")


# ─────────────────────────────────────────────
# 3c. CHART 3 — Heatmap: Correlation between numerical attributes
# ─────────────────────────────────────────────

# Compute Pearson correlation matrix for all four numerical columns
corr = df[numerical_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    corr,
    annot=True,           # Show correlation values inside each cell
    fmt=".2f",
    cmap="coolwarm",      # Diverging palette: blue = negative, red = positive
    center=0,
    linewidths=0.5,
    linecolor="white",
    square=True,
    ax=ax,
    annot_kws={"size": 10},
)

ax.set_title("Correlation Matrix — Numerical Attributes", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.close()
print("Saved → correlation_matrix.png")

print("\nAll done. Three PNG files have been generated.")
