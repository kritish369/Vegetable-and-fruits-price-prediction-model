import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import time

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("Todays_kalimati.csv", skiprows=1)

print(df.columns)

start_time = time.time()
print("The time model training started:", start_time)


# ==========================================
# 2. FILTER DATA
# ==========================================

# Only Apple (Jholey)
apple_df = df[df["Commodity"] == "Apple(Jholey)"].copy()

print("\n========== APPLE DATA ==========")
print(apple_df.describe())
print(apple_df.dtypes)


# ==========================================
# 3. CYCLIC FEATURE ENGINEERING
# ==========================================

# Month cycle
month_cycle_length = 12

angle_month = 2 * np.pi * (
    apple_df["Month"] / month_cycle_length
)

apple_df["Month_sin"] = np.sin(angle_month)
apple_df["Month_cos"] = np.cos(angle_month)


# Day-of-week cycle
week_cycle_length = 7

angle_week = 2 * np.pi * (
    apple_df["DayOfWeek"] / week_cycle_length
)

apple_df["Week_sin"] = np.sin(angle_week)
apple_df["Week_cos"] = np.cos(angle_week)


# ==========================================
# 4. SELECT FEATURES AND TARGET
# ==========================================

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


# ==========================================
# 5. CHECK MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")
print(X.isna().sum())


# ==========================================
# 6. REMOVE ROWS WITH MISSING FEATURES
# ==========================================

# Some early rows don't have enough historical
# data to calculate 7-day changes/volatility.

X = X.dropna()

# Keep y aligned with the remaining X rows
y = y.loc[X.index]

print("\n========== AFTER REMOVING NaN ==========")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Remaining NaN values:")
print(X.isna().sum())


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 8. CREATE MODEL
# ==========================================

model = LinearRegression()


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)


# ==========================================
# 10. MAKE PREDICTIONS
# ==========================================

pred_test_y = model.predict(X_test)


# ==========================================
# 11. DISPLAY PREDICTIONS
# ==========================================

print("\n========== PREDICTIONS ==========")

print("First 5 predictions:")
print(pred_test_y[:5])

print("\nFirst 5 actual values:")
print(y_test.iloc[:5].values)


# ==========================================
# 12. TRAINING TIME
# ==========================================

end_time = time.time()

print("\n========== TRAINING TIME ==========")
print("Total time:", end_time - start_time, "seconds")