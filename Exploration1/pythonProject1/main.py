# 1 install and import necessary libraries
import pandas as p
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plot

# 2 load .csv file into a pandas data frame
# set option to display entire columns
p.set_option("display.max_columns", None)

df = p.read_csv(r"C:\Users\aeren\Desktop\TEDU\ResearchProjectWithBengisenPekmen\Datasets\Data 1\Thyroid_Diff.csv")

print(df.head())

# 3 preprocess data
# categorical columns require encode
label_columns = [
    "Gender",
    "Smoking",
    "Hx Smoking",
    "Hx Radiothreapy",
    "Thyroid Function",
    "Physical Examination",
    "Adenopathy",
    "Pathology",
    "Focality",
    "Risk",
    "T", "N", "M",
    "Stage",
    "Response",
    "Recurred"
]

label_encoders = {}
for column in label_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

print(df.head())

# 4 split data into features and target
# define features x and target y
x = df.drop("Recurred", axis=1)
y = df["Recurred"]

# 5 split data into train and test sets
# utilize 80% for training 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 6 build decision tree model
# train a decision tree utilizing scikit_learn
model = DecisionTreeClassifier(random_state=42)

# train model
model.fit(x_train, y_train)

# predict about test data
y_predict = model.predict(x_test)

# 7 evaluate model
accuracy = metrics.accuracy_score(y_test, y_predict)
print(f"ACCURACY: {accuracy}")

# confusion matrix
cm = metrics.confusion_matrix(y_test, y_predict)
print(f"CONFUSION MATRIX:\n{cm}")

# 8 visualize decision tree model
plot.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=x.columns, class_names=["No", "Yes"])
plot.show()