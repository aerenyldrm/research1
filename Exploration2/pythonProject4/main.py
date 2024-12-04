import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense

# 1. Load dataset
data = pd.read_csv(r"C:\Users\aeren\Downloads\avv_v0.csv")

# 2. Split features and target
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target

# Reshape y if necessary
if len(y.shape) == 1:
    y = y.reshape(-1, 1)

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Construct ANN
model_ann = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dense(1, activation="linear")  # Regression output
])

# 6. Compile the model
model_ann.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 7. Train the model
history = model_ann.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=256, batch_size=64, verbose=1)

# 8. Predict and evaluate
predictions_ann = model_ann.predict(X_test)
mse_ann = mean_squared_error(y_test, predictions_ann)
r2_ann = r2_score(y_test, predictions_ann)

print(f"ANN PERFORMANCE\nMean Squared Error:\t{mse_ann}\nR2:\t{r2_ann}")
