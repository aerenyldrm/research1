import numpy as np
import pandas as pd
import random as rd
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# set seed for reproducibility
seed = 6
rd.seed(seed)
np.random.seed(seed)

data = pd.read_csv(r"C:\Users\aeren\Desktop\TEDU\ResearchProjectWithBengisenPekmen\Explorations\RESEARCH\INFUS2025\dataset\avv_variance_3040 - feature_inclusion.csv").values

# determine features and target
x = data[:, :-2] # features
y = data[:, -2] # target

print(x) # for debug
print(y) # for debug

# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# utilize random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=2)

# train model
model.fit(x_train, y_train)

# predict with test data
y_predict = model.predict(x_test)

# evaluate model
print(f" Decision Tree Mean Squared Error:\t{mean_squared_error(y_test, y_predict)}")

# visualize
plot.figure(figsize=(16, 9))
plot.scatter(y_test, y_predict, alpha=0.5, color="black")
plot.plot([y.min(), y.max()], [y.min(), y.max()], color="black")
plot.xlabel("Actual Values")
plot.ylabel("Predicted Values")
plot.title("Actual and Predicted Values Data")
plot.show()

plot.figure(figsize=(16, 9))
plot.scatter(y_test, y_test - y_predict, alpha=0.5, color="black")
plot.axhline(0, color="black")
plot.xlabel("Actual Values")
plot.ylabel("Residuals (Actual Values - Predicted Values")
plot.title("Residuals Data")
plot.show()

# feature importance
feature_importance = model.feature_importances_
importance_data_frame = pd.DataFrame({"Feature":["Re", "Gr", "Ha", "Rm"], "Importance": feature_importance}).sort_values(by="Importance", ascending=False)
plot.figure(figsize=(16, 9))
sns.barplot(x="Importance", y="Feature", data=importance_data_frame, color="black")
plot.title("Feature Importance")
plot.xlabel("Importance")
plot.ylabel("Feature")
plot.show()