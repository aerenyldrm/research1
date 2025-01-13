import random as rd
import keras as kr
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plot
from keras.api.layers import Dense
from keras.api.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NN:
    # initialize class with dataset path
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    @staticmethod
    def set_seeds(seed=2):
        # set seeds for reproducibility
        rd.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # for tensorflow and keras
        kr.api.utils.set_random_seed(seed)
        kr.utils.set_random_seed(seed)

        # additional tensorflow specific reproducibility settings
        tf.config.experimental.enable_op_determinism()

    # read dataset within path and understand it via python and numpy data structures
    def understand_dataset(self):
        data = pd.read_csv(self.dataset_path).values

        # shuffle data to prevent incorrect patterns
        indices_shuffle = np.random.permutation(len(data))
        data = data[indices_shuffle]

        # print(data) # for debug

        # determine features and target
        x = data[:, :-2] # features
        y = data[:, -2] # target

        # print(x) # for debug
        # print(y) # for debug

        # split dataset into train and test sets with percentage and random seed
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

        # normalize and standardize data x to prevent instability during training
        scaler_x = StandardScaler()

        # standardize features
        x_train = scaler_x.fit_transform(x_train)
        x_test = scaler_x.transform(x_test)

        return x_train, x_test, y_train, y_test, y

    @staticmethod
    def construct_ann(x_train, y_train):
        # construct ANN
        model_ann = Sequential([
            Dense(100, activation="relu"),
            Dense(100, activation="relu"),
            Dense(100, activation="relu"),
            Dense(1, activation="linear")
        ])

        # compile model
        model_ann.compile(optimizer="adam", loss="mse", metrics=["mse"])

        # train model
        model_ann.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1)

        return model_ann

    @staticmethod
    def predict_and_evaluate(model_ann, x_test, y_test):
        y_predict_ann = np.array(model_ann.predict(x_test)).flatten()
        mse_ann = mean_squared_error(y_test, y_predict_ann)

        print(f"ANN Mean Squared Error:\t{mse_ann}") # so, it does not return any value
        
        # print(y_predict_ann) # for debug
        # print(y_test) # for debug
        
        return y_test, y_predict_ann
    
    @staticmethod
    def plot(y, y_test, y_predict_ann):
        plot.figure(figsize=(16, 9))
        plot.scatter(y_test, y_predict_ann, alpha=0.5, color="black")
        plot.plot([y.min(), y.max()], [y.min(), y.max()], color="black")
        plot.xlabel("Actual Values")
        plot.ylabel("Predicted Values")
        plot.title("Actual and Predicted Values Data")
        plot.show()

        plot.figure(figsize=(16, 9))
        plot.scatter(y_test, y_test - y_predict_ann, alpha=0.5, color="black")
        plot.axhline(0, color="black")
        plot.xlabel("Actual Values")
        plot.ylabel("Residuals (Actual Values - Predicted Values")
        plot.title("Residuals Data")
        plot.show()

if __name__ == "__main__":
    # initialize object to utilize it
    object_NN = NN(r"C:\Users\aeren\Desktop\TEDU\ResearchProjectWithBengisenPekmen\Explorations\RESEARCH\INFUS2025\dataset\avv_variance_3040 - feature_inclusion.csv")

    # set seed exactly same for reproducibility
    object_NN.set_seeds(6)

    # understand attached dataset
    preprocessed_dataset = object_NN.understand_dataset()

    # come up with ANN
    particular_model_ann = object_NN.construct_ann(
        preprocessed_dataset[0],
        preprocessed_dataset[2]
    )

    # predict and evaluate constructed specific model
    result = object_NN.predict_and_evaluate(particular_model_ann, preprocessed_dataset[1], preprocessed_dataset[3])
    
    # display result
    object_NN.plot(preprocessed_dataset[4], result[0], result[1])