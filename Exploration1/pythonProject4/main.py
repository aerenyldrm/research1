# 1 import necessary libraries
import pandas as p
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
import matplotlib.pyplot as plot

# 2 load and preprocess data
df = p.read_csv(r"C:\Users\aeren\Desktop\TEDU\ResearchProjectWithBengisenPekmen\Datasets\Data 1\Thyroid_Diff.csv")

# encode categorical columns
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

# split features and target
x = df.drop("Recurred", axis=1)
y = df["Recurred"]

# split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 3 build neural network model
# initialize neural network
model = Sequential()

# input layer and first hidden layer with 32 neurons and ReLU activation
model.add(Dense(32, input_dim=x_train.shape[1], activation="relu"))

# second hidden layer with 8 neurons
model.add(Dense(8, activation="relu"))

# output layer with 1 neuron (binary classification) and sigmoid activation
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 4 train neural network
# train model
history = model.fit(x_train, y_train, epochs=53, batch_size=16, validation_split=0.2, verbose=1)

# 5 evaluate model
# evaluate on test set
y_predict = (model.predict(x_test) > 0.5).astype("int32")

# calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f"ACCURACY: {accuracy}")

# confusion matrix
cm = confusion_matrix(y_test, y_predict)
print(f"CONFUSION MATRIX:\n{cm}")

# 6 plot training history
# plot training and validation accuracy values
plot.figure(figsize=(12, 6))
plot.plot(history.history["accuracy"])
plot.plot(history.history["val_accuracy"])
plot.title("Model Accuracy")
plot.xlabel("Epoch")
plot.ylabel("Accuracy")
plot.legend(["Train", "Validation"],loc="upper left")
plot.show()

# plot training and validation loss values
plot.figure(figsize=(12, 6))
plot.plot(history.history["loss"])
plot.plot(history.history["val_loss"])
plot.title("Model Loss")
plot.xlabel("Epoch")
plot.ylabel("Loss")
plot.legend(["Train", "Validation"], loc="upper left")
plot.show()