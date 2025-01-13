import random as rd
import tensorflow as tf
import keras as kr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.api.models import Sequential
from keras.api.layers import Dense

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
        data = pd.read_csv(self.dataset_path)

        # determine features and target
        x = data.iloc[:, :-2].values # features
        y = data.iloc[:, -2].values # target

        # reshape y if necessary i.e., there is only 1 y
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # split dataset into train and test sets with percentage and random seed
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

        # normalize and standardize data x to prevent instability during training
        scaler_x = StandardScaler()

        # standardize features
        x_train = scaler_x.fit_transform(x_train)
        x_test = scaler_x.transform(x_test)

        """
        # convert back y_train and y_test back to 1d lists
        y_train = [item[0] for item in y_train]
        y_test = [item[0] for item in y_test]
        """

        return x_train, x_test, y_train, y_test

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
        predictions_ann = model_ann.predict(x_test)
        mse_ann = mean_squared_error(y_test, predictions_ann)

        print(f"ANN Mean Squared Error:\t{mse_ann}") # so, it does not return any value

if __name__ == "__main__":
    # initialize object to utilize it
    object_NN = NN(r"C:\Users\aeren\Desktop\TEDU\ResearchProjectWithBengisenPekmen\Explorations\Exploration4\INFUS2025\dataset\avv_variance_3040.csv")

    # set seed exactly same for reproducibility
    object_NN.set_seeds(5)

    # understand attached dataset
    preprocessed_dataset = object_NN.understand_dataset()

    # come up with ANN
    particular_model_ann = object_NN.construct_ann(
        preprocessed_dataset[0],
        preprocessed_dataset[2],
    )

    # predict and evaluate constructed specific model
    object_NN.predict_and_evaluate(particular_model_ann, preprocessed_dataset[1], preprocessed_dataset[3])