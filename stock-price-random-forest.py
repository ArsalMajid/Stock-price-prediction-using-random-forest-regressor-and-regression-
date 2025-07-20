import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------------------------------
# Step 1: Load the Enhanced Data & Convert Data Types
# -----------------------------------------------------------
df = pd.read_csv("daily_stock_prices_enhanced.csv")

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Function to convert volume strings (e.g., "1.28M", "911.68K") to float numbers
def convert_volume(vol):
    vol = str(vol).replace(",", "").strip()  # Remove commas and extra spaces
    if 'M' in vol:
        return float(vol.replace('M', '')) * 1_000_000
    elif 'K' in vol:
        return float(vol.replace('K', '')) * 1_000
    else:
        try:
            return float(vol)
        except ValueError:
            return None

# Convert the 'Vol.' column to numeric values if it exists
if "Vol." in df.columns:
    df["Vol."] = df["Vol."].apply(convert_volume)

# (Optional) Ensure 'High' and 'Low' columns are numeric
if "High" in df.columns:
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
if "Low" in df.columns:
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")

# Drop any rows with missing values that might have resulted from conversion or rolling calculations
df.dropna(inplace=True)

# -----------------------------------------------------------
# Step 2: Prepare Features & Target for Modeling
# -----------------------------------------------------------
# Expanded feature selection: adding High, Low, and volume ("Vol.") to our original features.
features = ["Prev_Price", "MA_7", "MA_30", "Volatility", "RSI", "High", "Low", "Vol."]
target = "Price"

# Split the data into training and testing sets
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------
# Step 3: Hyperparameter Tuning via Grid Search
# -----------------------------------------------------------
# Define a parameter grid to search over.
param_grid = {
    "n_estimators": [50, 100, 200],       # Number of trees in the forest
    "max_depth": [None, 10, 20],          # Maximum depth of the tree
    "min_samples_split": [2, 5, 10]       # Minimum number of samples required to split an internal node
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation and negative MSE as the scoring metric
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=5,
                           scoring="neg_mean_squared_error",
                           n_jobs=-1,
                           error_score='raise')  # Set error_score to 'raise' to debug if needed

# Run grid search on training data
grid_search.fit(X_train, y_train)

# Get and display the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# -----------------------------------------------------------
# Step 4: Train the Optimized Random Forest Model
# -----------------------------------------------------------
optimized_rf = RandomForestRegressor(**best_params, random_state=42)
optimized_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = optimized_rf.predict(X_test)

# Evaluate model performance using common regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# -----------------------------------------------------------
# Step 5: Visualize Predictions vs. Actual Prices
# -----------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test.values, label="Actual Prices", color="blue")
plt.plot(range(len(y_test)), y_pred, label="Predicted Prices", color="red", linestyle="dashed")
plt.title("Stock Price Prediction with Optimized Random Forest")
plt.xlabel("Test Sample Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()