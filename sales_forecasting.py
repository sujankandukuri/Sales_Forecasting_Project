import pandas as pd

# Load the sales dataset
data = pd.read_csv("dataset/train.csv")

# Show first 5 rows
print(data.head())

# Show column names
print("\nColumns:")
print(data.columns)
# Dataset information
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Basic statistics
print("\nSales Statistics:")
print(data["Sales"].describe())
# Convert Order Date to date format
data["Order Date"] = pd.to_datetime(data["Order Date"], dayfirst=True)

# Sort data by date
data = data.sort_values("Order Date")

print("\nDate range:")
print(data["Order Date"].min(), "to", data["Order Date"].max())

print("\nFirst 5 dates:")
print(data[["Order Date", "Sales"]].head())
import matplotlib.pyplot as plt

# Create daily sales
daily_sales = data.groupby("Order Date")["Sales"].sum()

# Plot sales trend
plt.figure(figsize=(12, 6))
plt.plot(daily_sales.index, daily_sales.values)

plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
# Create features from Order Date
data["Year"] = data["Order Date"].dt.year
data["Month"] = data["Order Date"].dt.month
data["Day"] = data["Order Date"].dt.day
data["DayOfWeek"] = data["Order Date"].dt.dayofweek

# Display the new features

print("\nNew Features:")
print(data[["Order Date", "Year", "Month", "Day", "DayOfWeek", "Sales"]].head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Select input features
X = data[["Year", "Month", "Day", "DayOfWeek"]]

# Target: Sales
y = data["Sales"]

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check model error
mae = mean_absolute_error(y_test, predictions)

print("\nModel Training Complete!")
print("Mean Absolute Error:", mae)

# Show some actual vs predicted values
results = pd.DataFrame({
    "Actual Sales": y_test.values[:10],
    "Predicted Sales": predictions[:10]
})

print("\nActual vs Predicted:")
print(results)
# Predict sales for a future date

future_date = pd.DataFrame({
    "Year": [2026],
    "Month": [11],
    "Day": [1],
    "DayOfWeek": [6]
})

future_prediction = model.predict(future_date)

print("\nFuture Sales Prediction:")
print("Predicted Sales:", future_prediction[0])
# Predict sales for the next 30 days

last_date = data["Order Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=30
)

future_data = pd.DataFrame({
    "Year": future_dates.year,
    "Month": future_dates.month,
    "Day": future_dates.day,
    "DayOfWeek": future_dates.dayofweek
})

future_predictions = model.predict(future_data)

forecast = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_predictions
})

print("\nNext 30 Days Sales Forecast:")
print(forecast)

print("\nTotal predicted sales for next 30 days:",
      forecast["Predicted Sales"].sum())
# Plot the 30-day sales forecast

plt.figure(figsize=(12, 6))

plt.plot(
    forecast["Date"],
    forecast["Predicted Sales"],
    marker="o"
)

plt.title("Next 30 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
from sklearn.metrics import r2_score

# Calculate R2 score
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
# Create and save 30-day forecast graph

plt.figure(figsize=(12, 6))

plt.plot(
    forecast["Date"],
    forecast["Predicted Sales"],
    marker="o"
)

plt.title("Next 30 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=45)

plt.tight_layout()

# Save the graph
plt.savefig("charts/30_day_forecast.png")

plt.show()
plt.savefig("charts/daily_sales_trend.png")
plt.show()
# Improved time-based model

features = ["Year", "Month", "Day", "DayOfWeek"]

data = data.sort_values("Order Date").reset_index(drop=True)

split_index = int(len(data) * 0.8)

train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

X_train = train_data[features]
y_train = train_data["Sales"]

X_test = test_data[features]
y_test = test_data["Sales"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nImproved Model Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)