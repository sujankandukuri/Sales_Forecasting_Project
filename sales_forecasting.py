import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset/train.csv")

print("Dataset loaded successfully!")
print(data.head())

# Convert Order Date to datetime
data["Order Date"] = pd.to_datetime(
    data["Order Date"],
    dayfirst=True
)

# Sort by date
data = data.sort_values("Order Date")

# Daily sales
daily_sales = data.groupby("Order Date")["Sales"].sum()

# Create chart
plt.figure(figsize=(14, 7))

plt.plot(
    daily_sales.index,
    daily_sales.values,
    linewidth=1.5
)

plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()

# Save chart
plt.savefig(
    "charts/daily_sales_trend.png",
    dpi=300,
    bbox_inches="tight"
)

# Display chart
plt.show()

print("Daily sales chart created successfully!")