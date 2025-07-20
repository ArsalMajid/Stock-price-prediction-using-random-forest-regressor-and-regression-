import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------
# Step 1: Load & Merge Stock Price Data
# -------------------------------------------

# Load the cleaned datasets from the current directory
df_march_april = pd.read_csv("daily_stock_prices_march-april2025.csv")
df_may_june = pd.read_csv("daily_stock_prices_may-june2025.csv")

# Merge the two datasets
df_daily = pd.concat([df_march_april, df_may_june], ignore_index=True)

# Convert 'Date' column to datetime format
df_daily["Date"] = pd.to_datetime(df_daily["Date"], format="%m/%d/%Y")

# Sort by date to ensure correct order
df_daily = df_daily.sort_values(by="Date", ascending=True)

# -------------------------------------------
# Step 2: Feature Engineering
# -------------------------------------------

# 1. Moving Averages (7-day and 30-day)
df_daily["MA_7"] = df_daily["Price"].rolling(window=7).mean()
df_daily["MA_30"] = df_daily["Price"].rolling(window=30).mean()

# 2. Volatility (Standard Deviation over 7 days)
df_daily["Volatility"] = df_daily["Price"].rolling(window=7).std()

# 3. Relative Strength Index (RSI) with 14-day window
delta = df_daily["Price"].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df_daily["RSI"] = 100 - (100 / (1 + rs))

# 4. Lag Feature (Previous Day's Price)
df_daily["Prev_Price"] = df_daily["Price"].shift(1)

# -------------------------------------------
# Step 3: Clean the Data (Remove NaN Rows)
# -------------------------------------------

# Drop rows that contain NaN values due to rolling calculations
df_daily.dropna(inplace=True)

# Save the enhanced dataset
df_daily.to_csv("daily_stock_prices_enhanced.csv", index=False)

# -------------------------------------------
# Step 4: Data Visualization
# -------------------------------------------

# Set a consistent style for plots
sns.set_style("darkgrid")

# 1️⃣ Stock Price & Moving Averages Plot
plt.figure(figsize=(12, 6))
sns.lineplot(x="Date", y="Price", data=df_daily, label="Stock Price", color="blue")
sns.lineplot(x="Date", y="MA_7", data=df_daily, label="7-Day MA", color="orange")
sns.lineplot(x="Date", y="MA_30", data=df_daily, label="30-Day MA", color="red")
plt.title("Stock Price & Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# 2️⃣ Volatility Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x="Date", y="Volatility", data=df_daily, color="purple", label="7-Day Volatility")
plt.title("Stock Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 3️⃣ Relative Strength Index (RSI) Plot
plt.figure(figsize=(12, 6))
sns.lineplot(x="Date", y="RSI", data=df_daily, color="green", label="RSI (14-Day)")
plt.axhline(y=70, color="red", linestyle="--", label="Overbought (Sell Signal)")
plt.axhline(y=30, color="blue", linestyle="--", label="Oversold (Buy Signal)")
plt.title("Relative Strength Index (RSI)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.xticks(rotation=45)
plt.legend()
plt.show()