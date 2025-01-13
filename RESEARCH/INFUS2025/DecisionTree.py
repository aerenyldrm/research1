import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
data_frame = pd.read_csv(r"C:\Users\aeren\Desktop\TEDU\ResearchProjectWithBengisenPekmen\Explorations\RESEARCH\INFUS2025\dataset\avv_variance_3040 - feature_inclusion.csv")

print(data_frame.head()) # for debug

# determine features and target
x = data_frame.drop(labels="avgA", axis=1).drop(labels="Navg", axis=1)
y = data_frame["Navg"]

print(x) # for debug
print(y) # for debug

# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# utilize decision tree regressor
model = DecisionTreeRegressor(random_state=2)

# train model
model.fit(x_train, y_train)

# predict with test data
y_predict = model.predict(x_test)

# evaluate model
print(f" Decision Tree Mean Squared Error:\t{mean_squared_error(y_test, y_predict)}")

# visualize tree
plot.figure(figsize=(16, 9))
plot_tree(model, feature_names=x.columns, max_depth=3, fontsize=6, filled=True)
plot.show()

# plots
plot.figure(figsize=(16, 9))
plot.scatter(y_test, y_predict, alpha=0.5, color="black")
plot.plot([y.min(), y.max()], [y.min(), y.max()], color="black")
plot.xlabel("Actual Values")
plot.ylabel("Predicted Values")
plot.title("Actual and Predicted Values Data")
plot.show()

residuals = y_test - y_predict
plot.figure(figsize=(16, 9))
plot.scatter(y_test, residuals, alpha=0.5, color="black")
plot.axhline(0, color="black")
plot.xlabel("Actual Values")
plot.ylabel("Residuals (Actual Values - Predicted Values")
plot.title("Residuals Data")
plot.show()

# feature importance
feature_importance = model.feature_importances_
importance_data_frame = pd.DataFrame({"Feature": x.columns, "Importance": feature_importance}).sort_values(by="Importance", ascending=False)
plot.figure(figsize=(16, 9))
sns.barplot(x="Importance", y="Feature", data=importance_data_frame, color="black")
plot.title("Feature Importance")
plot.xlabel("Importance")
plot.ylabel("Feature")
plot.show()