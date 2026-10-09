# Vegetable-and-fruits-price-prediction-model

# Kalimati Commodity Price Prediction

Predicts the next day's average price of each commodity in the Kalimati market dataset. For every commodity, the script trains five regression models and reports how well each one performs on training and test data.

## How it works

1. **Load the data** from `Todays_kalimati.csv` (the first row is skipped).
2. **Engineer cyclic features.** Month and day of week are converted to sine/cosine pairs so the models understand that December is close to January and Sunday is close to Monday.
3. **Loop over each commodity.** For each one, the script:
   - filters the data to that commodity,
   - replaces `inf` values with NaN and drops any row with a missing feature or target,
   - skips the commodity if fewer than 30 usable rows remain,
   - splits the data into 80% training and 20% test sets,
   - trains five fresh models on that commodity only,
   - prints training time and error metrics for each model.
4. **Print the total time** taken for the whole run.

## Features and target

| Role | Columns |
|------|---------|
| Features | `Today_Avg`, `Month_sin`, `Month_cos`, `Week_sin`, `Week_cos`, `Year`, `Previous_Day_Avg`, `7_Day_Avg`, `Price_Change_7_Day`, `Price_Change_1_Day`, `Price_Volatility` |
| Target | `Next_Day_Avg` |

## Models

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors

## Metrics reported

For each commodity and model:

- Training time (seconds)
- Test: MAE, RMSE, MSE
- Train: MAE, RMSE, MSE

Comparing train and test RMSE shows overfitting. A much lower train error than test error means the model is memorizing the training data.

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn

Install with:

```bash
pip install pandas numpy scikit-learn
```

## Usage

1. Place `Todays_kalimati.csv` in the same folder as the script.
2. Run:

```bash
python food_veg_price.py
```

## Example output

```
Commodity: Tomato Big(Nepali), Model: Random Forest
Training Time: 0.4123 seconds
Mean Absolute Error (MAE): 2.1000
Root Mean Squared Error (RMSE): 3.0000
Mean Squared Error (MSE): 9.0000
Training RMSE: 1.2000
Training MSE: 1.4400
Training MAE: 0.8000
--------------------------------------------------
```

The numbers above are illustrative only.

## Notes and limitations

- **Random split.** The train/test split is shuffled. Because the features and target are time-based, the training set can contain days that come after test days, which makes test scores look better than they would be on genuinely future data. A chronological split (train on earlier dates, test on later ones) gives a more realistic estimate.
- **KNN and scaling.** K-Nearest Neighbors is sensitive to feature scale. Scaling features with `StandardScaler` would likely improve it.
- **Reproducibility.** The Decision Tree, Random Forest and Gradient Boosting models have no `random_state`, so results can vary slightly between runs.
- **Skipped commodities.** Commodities with fewer than 30 usable rows are skipped, and a message is printed for each one.

## Possible improvements

- Chronological train/test split
- Feature scaling for KNN
- Add a naive baseline (predict tomorrow = today) to check the models add value
- Save results to a CSV for comparison across commodities
- Add RAE (relative absolute error) alongside MAE and RMSE
