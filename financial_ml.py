"""
Financial Analytics — Linear Regression with Scikit-Learn
Dataset : Financial-Analytics-data1.csv
Feature (X) : sales_qtr_clean  (quarterly sales, ₹ crore)
Target  (y) : market_cap_clean (market capitalisation, ₹ crore)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN THE DATASET
# ─────────────────────────────────────────────────────────────

# Load raw CSV
df = pd.read_csv("Financial-Analytics-data1.csv")

# Strip commas from comma-formatted number strings and convert to float.
# The result is stored with the "_clean" suffix to match the task spec.
df["sales_qtr_clean"] = (
    df["sales_qtr_crore"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace("nan", np.nan)
    .pipe(pd.to_numeric, errors="coerce")
)

df["market_cap_clean"] = (
    df["market_cap_crore"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace("nan", np.nan)
    .pipe(pd.to_numeric, errors="coerce")
)

# Drop rows where either of the two modelling columns is missing
df.dropna(subset=["sales_qtr_clean", "market_cap_clean"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Records available for modelling: {len(df)}")

# ─────────────────────────────────────────────────────────────
# 2. PREPARE FEATURES AND TARGET
# ─────────────────────────────────────────────────────────────

# X must be 2-D for scikit-learn; reshape (-1,1) turns the Series
# into a column vector of shape (n_samples, 1).
X = df["sales_qtr_clean"].values.reshape(-1, 1)   # input feature
y = df["market_cap_clean"].values                  # target variable

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (80 % train — 20 % test)
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN LINEAR REGRESSION MODEL
# ─────────────────────────────────────────────────────────────

model = LinearRegression()
model.fit(X_train, y_train)

# Learned parameters
print(f"\nModel coefficient (slope)    : {model.coef_[0]:,.4f}")
print(f"Model intercept              : {model.intercept_:,.4f}")

# ─────────────────────────────────────────────────────────────
# 5. EVALUATE ON THE TEST SET
# ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 45)
print("MODEL EVALUATION (Test Set)")
print("=" * 45)
print(f"  R² Score : {r2:.4f}  ({r2 * 100:.2f}% variance explained)")
print(f"  RMSE     : ₹ {rmse:,.2f} crore")
print("=" * 45)

# ─────────────────────────────────────────────────────────────
# 6. SCATTER PLOT + REGRESSION LINE  →  regression_plot.png
# ─────────────────────────────────────────────────────────────

# Build a smooth regression line across the full range of X_test
x_line = np.linspace(X_test.min(), X_test.max(), 300).reshape(-1, 1)
y_line = model.predict(x_line)

fig, ax = plt.subplots(figsize=(10, 6))

# --- Actual test points ---
ax.scatter(
    X_test, y_test,
    color="#3b82d4",
    alpha=0.65,
    edgecolors="white",
    linewidths=0.5,
    s=55,
    label="Actual (test set)",
    zorder=3,
)

# --- Predicted points (residual connection, optional visual aid) ---
ax.scatter(
    X_test, y_pred,
    color="#f97316",
    alpha=0.45,
    marker="x",
    s=40,
    label="Predicted",
    zorder=3,
)

# --- Fitted regression line ---
ax.plot(
    x_line, y_line,
    color="#dc2626",
    linewidth=2,
    label=(
        f"Regression line\n"
        f"y = {model.coef_[0]:,.1f}·x + {model.intercept_:,.0f}\n"
        f"R² = {r2:.4f}  |  RMSE = {rmse:,.0f}"
    ),
    zorder=4,
)

# ── Formatting ───────────────────────────────
ax.set_xlabel("Quarterly Sales (₹ crore)", fontsize=12)
ax.set_ylabel("Market Capitalisation (₹ crore)", fontsize=12)
ax.set_title(
    "Linear Regression: Market Cap vs Quarterly Sales",
    fontsize=14, fontweight="bold", pad=14,
)

# Human-readable tick labels (e.g. 1,00,000)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

ax.legend(fontsize=9, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.savefig("regression_plot.png", dpi=150)
plt.close()

print("\nSaved → regression_plot.png")
print("Done.")
