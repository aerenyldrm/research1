# 1 install and import necessary libraries
import pandas as p
from PIL.ImageColor import colormap
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as pyp
import numpy as n

# 2 load dataset
# set option to display entire columns
p.set_option("display.max_columns", None)

df = p.read_csv (r"C:\Users\aeren\Downloads\avv_v0.csv")

# 3 separate features and target
x = df.drop(df.columns[-1], axis=1)
y = df[df.columns[-1]]

print(x)
print(y)

# 4 split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5 train a decision tree
model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(x_train, y_train)

# 6 predict and evaluate
y_prediction = model.predict(x_test)
mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)

print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")

# 7 scatter plot of actual vs predicted values
pyp.figure(figsize=(7,5))
pyp.scatter(y_test, y_prediction, alpha=0.5, color="black")
pyp.plot([y.min(), y.max()], [y.min(), y.max()], color="black")
pyp.xlabel("Actual Values")
pyp.ylabel("Predicted Values")
pyp.title("Actual and Predicted Values Data")
pyp.show()

# 8 residuals plot
residuals = y_test - y_prediction
pyp.figure(figsize=(7, 5))
pyp.scatter(y_test, residuals, alpha=0.5, color="black")
pyp.axhline(0, color="black")
pyp.xlabel("Actual Values")
pyp.ylabel("Residuals (Actual Values - Predicted Values")
pyp.title("Residuals Data")
pyp.show()

# 9 feature importance
feature_importance = model.feature_importances_
pyp.figure(figsize=(7,5))
pyp.barh(x.columns, feature_importance, color="black")
pyp.xlabel("Feature Importance")
pyp.title("Feature Importance for Decision Tree")
pyp.show()