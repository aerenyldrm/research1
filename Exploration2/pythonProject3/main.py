import pandas as p
import numpy as n
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost.dask import predict
import matplotlib.pyplot as plot
from xgboost import plot_importance

# 1 load dataset from .csv file
data = p.read_csv(r"C:\Users\aeren\Downloads\avv.csv")

# 2 understand dataset
print(data.head(7)) # preview first 7 rows
print(f"{data.info()}\n") # dataset information

# 3 determine features and target
x = data.iloc[:, :-2] # features
y = data.iloc[:, -2] # target

# 4 split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5 common function to train and evaluate boost models
def train_and_evaluate(model, model_name, x_for_train, y_for_train, x_for_test, y_for_test):
    # train model
    model.fit(x_for_train, y_for_train)

    # come up with predictions
    predictions = model.predict(x_for_test)

    # calculate evaluation metrics
    mse = mean_squared_error(y_for_test, predictions)
    r2 = r2_score(y_for_test, predictions)
    residuals = y_for_test - predictions

    # print performance
    print(f"{model_name} PERFORMANCE\nMean Squared Error:\t{mse}\nR2:\t{r2}\n")
    return model, mse, r2, predictions, residuals

# 6 boosts and commonly utilize above defined function to implement those
# Gradient Boost
gb_model = GradientBoostingRegressor(n_estimators=256, learning_rate=0.7, max_depth=7, random_state=42)
gb_model, gb_mse, gb_r2, gb_predictions, gb_residuals = train_and_evaluate(gb_model, "Gradient Boost", x_train, y_train, x_test, y_test)

# XGBoost
xgb_model = XGBRegressor(n_estimators=256, learning_rate=0.7, max_depth=7, random_state=42)
xgb_model, xgb_mse, xgb_r2, xgb_predictions, xgb_residuals = train_and_evaluate(xgb_model, "XGBoost", x_train, y_train, x_test, y_test)

# LightGBM
lgbm_model = LGBMRegressor(n_estimators=256, learning_rate=0.7, max_depth=7, random_state=42)
lgbm_model, lgbm_mse, lgbm_r2, lgbm_predictions, lgbm_residuals = train_and_evaluate(lgbm_model, "LightGBM", x_train, y_train, x_test, y_test)

# CatBoost
cat_model = CatBoostRegressor(iterations=256, learning_rate=0.7, depth=7, random_state=42)
cat_model, cat_mse, cat_r2, cat_predictions, cat_residuals = train_and_evaluate(cat_model, "CatBoost", x_train, y_train, x_test, y_test)

# 7 plot XGBoost results since its low mean squared error
plot.figure(figsize=(7, 5))
# predicted in comparison to actual data
plot.scatter(y_test, xgb_predictions, alpha=0.5, color="black")
plot.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black")
plot.title("Predicted and Actual Values")
plot.xlabel("Actual Values")
plot.ylabel("Predicted Values")
plot.show()

# residuals plot
plot.scatter(xgb_predictions, xgb_residuals, alpha=0.5, color="black")
plot.axhline(y=0, color='black')
plot.title("Residual Plot")
plot.xlabel("Predicted Values")
plot.ylabel("Residuals (Actual Value - Predicted Value)")
plot.show()

# feature importance plot
plot_importance(xgb_model, importance_type="weight")
plot.title("XGBoost Feature Importance")
plot.show()