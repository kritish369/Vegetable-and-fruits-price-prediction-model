from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from time import time 
# Reading the csv and initializing the dataframe

df = pd.read_csv("Todays_kalimati.csv", skiprows=1)
# cyclic feature engineering for month and day of week
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
df["Week_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
df["Week_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)   

# Feature and target columns
features = [
    "Today_Avg", "Month_sin", "Month_cos", "Week_sin", "Week_cos", "Year",
    "Previous_Day_Avg", "7_Day_Avg", "Price_Change_7_Day",
    "Price_Change_1_Day", "Price_Volatility"
]
target = "Next_Day_Avg"
  
# See which columns have missing values
print(df[features + [target]].isna().sum())
print("-" * 50)
# time 
start_time = time()
print(f"Model training started at: {start_time}")

# Loop through each unique commodity in the dataset
for commodity in df["Commodity"].unique():
    # Filter the dataframe for the current commodity
    commodity_df = df[df["Commodity"] == commodity].copy()
     # Turn inf into NaN, then drop rows missing any feature or the target
    commodity_df = commodity_df.replace([np.inf, -np.inf], np.nan)
    commodity_df = commodity_df.dropna(subset=features + [target])
    
        # Skip commodities with too little data
    if len(commodity_df) < 30:
        print(f"Skipping {commodity}: only {len(commodity_df)} usable rows")
        continue
    
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        commodity_df[features],
        commodity_df[target],
        test_size=0.2,
        random_state=42
        )
    # Initialize the models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "K-Nearest Neighbors": KNeighborsRegressor()
    }
    # Loop through each model and train it on the training data
    for model_name, model in models.items():
        start_time = time()
        model.fit(X_train, y_train)
        end_time = time()
        training_time = end_time - start_time
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        # Make predictions on the training set
        y_train_pred = model.predict(X_train)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        # Print the results for the current commodity and model
        print(f"Commodity: {commodity}, Model: {model_name}")
        print(f"Training Time: {training_time:.4f} seconds")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print("-" * 50)
# Print the total time taken for model training
end_time = time()
total_time = end_time - start_time
print(f"Total time taken for model training: {total_time:.4f} seconds")