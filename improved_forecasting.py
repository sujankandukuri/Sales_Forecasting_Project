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

# Convert individual sales records into daily sales
daily_sales = (
    data.groupby("Order Date")["Sales"]
    .sum()
    .reset_index()
)

# Create time-based features
daily_sales["Year"] = daily_sales["Order Date"].dt.year
daily_sales["Month"] = daily_sales["Order Date"].dt.month
daily_sales["Day"] = daily_sales["Order Date"].dt.day
daily_sales["DayOfWeek"] = daily_sales["Order Date"].dt.dayofweek

# Create previous-sales features
daily_sales["Lag1"] = daily_sales["Sales"].shift(1)
daily_sales["Lag7"] = daily_sales["Sales"].shift(7)
daily_sales["Lag30"] = daily_sales["Sales"].shift(30)

# Remove rows with missing lag values
daily_sales = daily_sales.dropna().reset_index(drop=True)

# Features
features = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Lag1",
    "Lag7",
    "Lag30"
]

X = daily_sales[features]
y = daily_sales["Sales"]

# Time-based train/test split
split_index = int(len(daily_sales) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Create model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== IMPROVED FORECASTING MODEL =====")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Compare actual and predicted values
results = pd.DataFrame({
    "Date": daily_sales["Order Date"].iloc[split_index:],
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

print("\nActual vs Predicted:")
print(results.head(10))

# Plot actual vs predicted
plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual Sales"],
    label="Actual Sales"
)

plt.plot(
    results["Date"],
    results["Predicted Sales"],
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("charts/actual_vs_predicted.png")

plt.show()