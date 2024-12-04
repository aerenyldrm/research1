import pandas as p
import numpy as n
from numpy.ma.core import indices
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plot

# 1 load dataset
df = p.read_csv(r"C:\Users\aeren\Downloads\avv_v0.csv")

# 2 understand data
print(df.head())
print(df.info())

# 3 determine features and target
x = df.iloc[:, :-1] # features
y = df.iloc[:, -1] # target

# 4 split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5 train a random forest regressor
rf_model = RandomForestRegressor(n_estimators=553, random_state=42)
rf_model.fit(x_train, y_train)

# 6 predict and evaluate random forest
rf_predictions = rf_model.predict(x_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print(f"MEAN SQUARED ERROR:\t{rf_mse}\nMEAN ABSOLUTE ERROR:\t{rf_mae}")

# 7 calculate residuals
residuals = y_test - rf_predictions

# 8 visualize results
# predicted in comparison to actual plot
plot.figure(figsize=(7, 5))
plot.scatter(y_test, rf_predictions, alpha=0.5, color="black")
plot.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black")
plot.title("Predicted and Actual Values Data")
plot.xlabel("Actual Values")
plot.ylabel("Predicted Values")
plot.show()

# residuals plot
plot.figure(figsize=(7, 5))
plot.scatter(y_test, residuals, alpha=0.5, color="black")
plot.axhline(0, color="black")
plot.title("Residuals Data")
plot.xlabel("Predicted Values")
plot.ylabel("Residuals (Actual Values - Predicted Values)")
plot.show()

# feature importance plot
importance_list = rf_model.feature_importances_
index_list = n.argsort(importance_list[::-1])
plot.figure(figsize=(7, 5))
plot.title("Feature Importance Data for Random Forest")
plot.bar(range(x.shape[1]), importance_list[index_list], align="center", color="black")
plot.xticks(range(x.shape[1]), x.columns[index_list], rotation=90)
plot.tight_layout()
plot.show()