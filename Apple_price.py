import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import time

df = pd.read_csv("Todays_kalimati.csv",skiprows=1)
print(df.columns)
start_time = time.time()
print("The time model trainning started:",start_time)
# Data Filtering and Data engineering (Feature engineering)
# for commodity Apple(Jholey)only
tomato_df = df[df["Commodity"]=="Apple(Jholey)"].copy()
print(tomato_df.describe(),tomato_df.dtypes)
# Cyclic feature structuring and design encoding for week and months 
month_cycle_length = 12
week_cycle_length = 7 
angle_month = 2 * np.pi * (tomato_df["Month"]/month_cycle_length)
tomato_df["Month_sin"] = np.sin(angle_month)
tomato_df["Month_cos"] = np.cos(angle_month)
angle_week = 2 * np.pi * (tomato_df["DayOfWeek"]/week_cycle_length)
tomato_df["Week_sin"] = np.sin(angle_week)
tomato_df["Week_cos"] = np.cos(angle_week)
model = LinearRegression()

X = tomato_df[["Today_Avg","Month_sin","Month_cos","Week_sin","Week_cos","Year","Previous_Day_Avg","7_Day_Avg","Price_Change_7_Day","Price_Change_1_Day","Price_Volatility"]]
y = tomato_df["Next_Day_Avg"]
print(X.isna().sum())
"""X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state =42)
model.fit(X_train,y_train)
pred_test_y = model.predict(X_test)
print("First 5 prediction of test x : ",pred_test_y[:5])
print("First 5 result of test x : ",y_test[:5])
"""




