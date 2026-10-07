import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import time

#Reading the dataset
df = pd.read_csv("Todays_kalimati.csv", skiprows=1)


start_time = time.time()
print("The time model training started:", start_time)
# Data filtering for Apple (Jholey) and data engineering
apple_df = df[df["Commodity"] == "Apple(Jholey)"].copy()

# Cyclic encoding design for month and day of week
month_cycle_length = 12
week_cycle_length = 7
apple_df["Month_sin"] = np.sin(2 * np.pi * (apple_df["Month"] / month_cycle_length))
apple_df["Month_cos"] = np.cos(2 * np.pi * (apple_df["Month"] / month_cycle_length))
apple_df["Week_sin"] = np.sin(2 * np.pi * (apple_df["DayOfWeek"] / week_cycle_length))
apple_df["Week_cos"] = np.cos(2 * np.pi * (apple_df["DayOfWeek"] / week_cycle_length))

#Initializing features(X) and target(y)
X = apple_df[
    [
        "Today_Avg",
        "Month_sin",
        "Month_cos",
        "Week_sin",
        "Week_cos",
        "Year",
        "Previous_Day_Avg",
        "7_Day_Avg",
        "Price_Change_7_Day",
        "Price_Change_1_Day",
        "Price_Volatility"
    ]
]

y = apple_df["Next_Day_Avg"]

print("\n========== FINAL NaN CHECK ==========")
print("X NaN:")
print(X.isna().sum())

print("\ny NaN:")
print(y.isna().sum())

print("\nTotal X NaN:", X.isna().sum().sum())
print("Total y NaN:", y.isna().sum())
# ==========================================
# REMOVE MISSING VALUES
# ==========================================

data = pd.concat([X, y], axis=1).dropna()

X = data.drop(columns=["Next_Day_Avg"])
y = data["Next_Day_Avg"]

print("\n========== AFTER REMOVING NaN ==========")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Total X NaN:", X.isna().sum().sum())
print("Total y NaN:", y.isna().sum())

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the model
linear_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor()
gradient_boosting_model = GradientBoostingRegressor()
knn_model = KNeighborsRegressor()

# traning the model 
# Linear Regression 
linear_model.fit(X_train, y_train)
pred_test_y = linear_model.predict(X_test)
pred_train_y = linear_model.predict(X_train)
print("Linear Regression model trained successfully.")
print("error metrics for Linear Regression model:")
print("MSE:", mean_squared_error(y_test, pred_test_y))
print("RMSE:", root_mean_squared_error(y_test, pred_test_y))
print("MAE:", mean_absolute_error(y_test, pred_test_y))
print("Training MSE:", mean_squared_error(y_train, pred_train_y))
print("Training RMSE:", root_mean_squared_error(y_train, pred_train_y))
print("Training MAE:", mean_absolute_error(y_train, pred_train_y))

# Decision Tree Regressor
decision_tree_model.fit(X_train, y_train)
pred_test_y = decision_tree_model.predict(X_test)
pred_train_y = decision_tree_model.predict(X_train)
print("\nDecision Tree Regressor model trained successfully.")
print("error metrics for Decision Tree Regressor model:")
print("MSE:", mean_squared_error(y_test, pred_test_y))
print("RMSE:", root_mean_squared_error(y_test, pred_test_y))
print("MAE:", mean_absolute_error(y_test, pred_test_y))
print("Training MSE:", mean_squared_error(y_train, pred_train_y))
print("Training RMSE:", root_mean_squared_error(y_train, pred_train_y))
print("Training MAE:", mean_absolute_error(y_train, pred_train_y))

# Random Forest Regressor
random_forest_model.fit(X_train, y_train)
pred_test_y = random_forest_model.predict(X_test)
pred_train_y = random_forest_model.predict(X_train)
print("\nRandom Forest Regressor model trained successfully.")
print("error metrics for Random Forest Regressor model:")
print("MSE:", mean_squared_error(y_test, pred_test_y))
print("RMSE:", root_mean_squared_error(y_test, pred_test_y))            
print("MAE:", mean_absolute_error(y_test, pred_test_y))
print("Training MSE:", mean_squared_error(y_train, pred_train_y))
print("Training RMSE:", root_mean_squared_error(y_train, pred_train_y))
print("Training MAE:", mean_absolute_error(y_train, pred_train_y))

# Gradient Boosting Regressor
gradient_boosting_model.fit(X_train, y_train)
pred_test_y = gradient_boosting_model.predict(X_test)
pred_train_y = gradient_boosting_model.predict(X_train)
print("\nGradient Boosting Regressor model trained successfully.")
print("error metrics for Gradient Boosting Regressor model:")
print("MSE:", mean_squared_error(y_test, pred_test_y))
print("RMSE:", root_mean_squared_error(y_test, pred_test_y))
print("MAE:", mean_absolute_error(y_test, pred_test_y))
print("Training MSE:", mean_squared_error(y_train, pred_train_y))
print("Training RMSE:", root_mean_squared_error(y_train, pred_train_y))
print("Training MAE:", mean_absolute_error(y_train, pred_train_y))  

# K-Nearest Neighbors Regressor
knn_model.fit(X_train, y_train)     
pred_test_y = knn_model.predict(X_test)
pred_train_y = knn_model.predict(X_train)
print("\nK-Nearest Neighbors Regressor model trained successfully.")
print("error metrics for K-Nearest Neighbors Regressor model:")
print("MSE:", mean_squared_error(y_test, pred_test_y))
print("RMSE:", root_mean_squared_error(y_test, pred_test_y))
print("MAE:", mean_absolute_error(y_test, pred_test_y))
print("Training MSE:", mean_squared_error(y_train, pred_train_y))
print("Training RMSE:", root_mean_squared_error(y_train, pred_train_y))
print("Training MAE:", mean_absolute_error(y_train, pred_train_y))
