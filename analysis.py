# Retail Data Set Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

%matplotlib inline

data_path = "retail_sales_dataset.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])

df = df.dropna(subset=["Transaction ID", "Date", "Total Amount"]).drop_duplicates()
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

print("Shape:", df.shape)
print(df.dtypes)
print("\nMissing per column:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

# Sales Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["Total Amount"], bins=50, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Total Amount ($)")
plt.show()

# Top 10 Product Categories by Total Sales
top_categories = df.groupby("Product Category")["Total Amount"].sum().nlargest(10)
plt.figure(figsize=(8, 5))
top_categories.plot(kind="barh")
plt.title("Top 10 Product Categories by Total Sales")
plt.xlabel("Total Amount ($)")
plt.gca().invert_yaxis()
plt.show()

# Monthly Revenue Over Time
monthly_rev = df.set_index("Date")["Total Amount"].resample("M").sum()
plt.figure(figsize=(10, 4))
monthly_rev.plot(marker="o")
plt.title("Monthly Revenue Over Time")
plt.ylabel("Total Amount ($)")
plt.show()

# Monthly Sales by Gender, stacked by Product Category
df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
df["MonthName"] = df["Date"].dt.strftime("%B")
pivot = df.pivot_table(index=["YearMonth", "Gender", "MonthName"], columns="Product Category", values="Total Amount", aggfunc="sum", fill_value=0, observed=True)
genders = pivot.index.get_level_values("Gender").unique()
fig, axes = plt.subplots(1, len(genders), figsize=(14, 6), sharey=True)
axes = [axes] if len(genders) == 1 else axes
for ax, gen in zip(axes, genders):
    gender_df = pivot.xs(gen, level="Gender").sort_index(level="YearMonth")
    month_names = gender_df.index.get_level_values("MonthName")
    gender_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c", linewidth=0, alpha=0.85)
    ax.set_title(f"{gen} Sales by Month")
    ax.set_xlabel("Month")
    ax.set_xticklabels(month_names, rotation=45)
    ax.set_ylabel("Total Sales ($)" if ax is axes[0] else "")
    ax.legend().set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Product Category", bbox_to_anchor=(1.02, 0.5), loc="center left")
plt.tight_layout()
plt.show()

# Correlations
numeric_cols = ["Age", "Quantity", "Price per Unit", "Total Amount", "Year", "Month"]
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()

# Customer Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.show()

# Model Training and Prediction
model_df = df.dropna(subset=["Age", "Quantity", "Price per Unit"])
X = model_df[["Age", "Quantity", "Price per Unit"]]
y = model_df["Total Amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Actual vs Predicted Total Amount (Sample of 100)
cmp = pd.DataFrame({"Actual": y_test.reset_index(drop=True), "Predicted": y_pred}).iloc[:100]
plt.figure(figsize=(12, 6))
plt.plot(cmp.index, cmp["Actual"], label="Actual", linewidth=2, marker="o", markersize=4)
plt.plot(cmp.index, cmp["Predicted"], label="Predicted", linewidth=2, linestyle="--", marker="o", markersize=4)
plt.grid(True, linestyle="--", alpha=0.5)
plt.title("Actual vs Predicted Total Amount (Sample of 100)")
plt.xlabel("Sample Index")
plt.ylabel("Total Amount")
plt.legend()
plt.tight_layout()
plt.show()

# Monthly Actual vs Target Total Amount
monthly_actual = df.set_index("Date")["Total Amount"].resample("M").sum()
np.random.seed(42)
mean_val = monthly_actual.mean()
monthly_target = pd.Series(mean_val * (1 + np.random.uniform(-0.1, 0.1, size=len(monthly_actual))), index=monthly_actual.index, name="Target")
comp = pd.DataFrame({"Actual": monthly_actual, "Target": monthly_target})
comp.index = comp.index.strftime("%B")
plt.figure(figsize=(12, 6))
sns.lineplot(data=comp, markers=True, dashes=False)
plt.title("Monthly Actual vs Target Total Amount")
plt.xlabel("Month")
plt.ylabel("Total Amount ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average Daily Transactional Revenue
df["Day of Week"] = df["Date"].dt.day_name()
sales_by_dow = df.groupby("Day of Week", observed=True)["Total Amount"].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.figure(figsize=(10, 6))
plt.plot(sales_by_dow.index, sales_by_dow.values, marker="o", linewidth=2)
plt.xlabel("Day")
plt.ylabel("Average Sales ($)")
plt.title("Average Daily Transactional Revenue")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
