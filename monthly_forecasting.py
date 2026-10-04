import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("dataset/train.csv")

# Convert date
data["Order Date"] = pd.to_datetime(
    data["Order Date"],
    dayfirst=True
)

# Sort by date
data = data.sort_values("Order Date")

# Create monthly sales
monthly_sales = (
    data.set_index("Order Date")["Sales"]
    .resample("ME")
    .sum()
    .reset_index()
)

# Create time features
monthly_sales["Year"] = monthly_sales["Order Date"].dt.year
monthly_sales["Month"] = monthly_sales["Order Date"].dt.month

# Create previous-month features
monthly_sales["Lag1"] = monthly_sales["Sales"].shift(1)
monthly_sales["Lag2"] = monthly_sales["Sales"].shift(2)
monthly_sales["Lag3"] = monthly_sales["Sales"].shift(3)

# Remove missing values
monthly_sales = monthly_sales.dropna().reset_index(drop=True)

# Features
features = [
    "Year",
    "Month",
    "Lag1",
    "Lag2",
    "Lag3"
]

X = monthly_sales[features]
y = monthly_sales["Sales"]

# Time-based split
split_index = int(len(monthly_sales) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MONTHLY FORECASTING MODEL =====")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Actual vs predicted
results = pd.DataFrame({
    "Date": monthly_sales["Order Date"].iloc[split_index:],
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

print("\nActual vs Predicted:")
print(results)

# Graph
plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    results["Date"],
    results["Predicted Sales"],
    marker="o",
    label="Predicted Sales"
)

plt.title("Monthly Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("charts/monthly_actual_vs_predicted.png")

plt.show()
# Business Insights

highest_month = monthly_sales.loc[
    monthly_sales["Sales"].idxmax()
]

lowest_month = monthly_sales.loc[
    monthly_sales["Sales"].idxmin()
]

average_monthly_sales = monthly_sales["Sales"].mean()

print("\n===== BUSINESS INSIGHTS =====")

print(
    "Highest Sales Month:",
    highest_month["Order Date"].strftime("%B %Y")
)

print(
    "Highest Sales:",
    highest_month["Sales"]
)

print(
    "Lowest Sales Month:",
    lowest_month["Order Date"].strftime("%B %Y")
)

print(
    "Lowest Sales:",
    lowest_month["Sales"]
)

print(
    "Average Monthly Sales:",
    average_monthly_sales
)