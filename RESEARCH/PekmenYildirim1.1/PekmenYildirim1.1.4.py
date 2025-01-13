import keras as kr
import numpy as np
import random as rd
import pandas as pd
import tensorflow as tf
import matlab.engine as me
from keras.api.layers import Dense
from keras.api.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NN:
    # initialize class with dataset path
    def __init__(self):
        self.matlab = me.start_matlab()

    @staticmethod
    def set_seeds(seed):
        # set seeds for reproducibility
        rd.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # for tensorflow and keras
        kr.api.utils.set_random_seed(seed)
        kr.utils.set_random_seed(seed)

        # additional tensorflow specific reproducibility settings
        tf.config.experimental.enable_op_determinism()

    def generate_matlab_dataset(self, start, end, step):
        # produce initial parameters send these to MATLAB for calculations
        parameter_set = [float(x) for x in list(np.linspace(start, end, step))]

        # generate initial dataset with list comprehension
        row_form_of_dataset = [[parameter, self.matlab.sqrt(parameter)] for parameter in parameter_set]
        dataset_in = np.array([value[0] for value in row_form_of_dataset])
        dataset_out = np.array([value[1] for value in row_form_of_dataset])

        # shuffle data to prevent incorrect patterns
        indices_shuffle = np.random.permutation(len(dataset_in))
        dataset_in = dataset_in[indices_shuffle]
        dataset_out = dataset_out[indices_shuffle]

        return dataset_in, dataset_out

    # read dataset within path and understand it via python and numpy data structures
    @staticmethod
    def understand_dataset(dataset_in, dataset_out):
        # determine features and target
        x = np.array(dataset_in).reshape(-1, 1)
        y = np.array(dataset_out).reshape(-1, 1)

        # reshape y if necessary i.e., there is only 1 y
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # split dataset into train and test sets with percentage and random seed
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

        # normalize and standardize data x to prevent instability during training
        scaler_x = StandardScaler()
        x_train = scaler_x.fit_transform(x_train)
        x_test = scaler_x.transform(x_test)

        # optionally scale outputs, y, only if it is necessary; comment out following if unnecessary
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def construct_ann(x_train, y_train):
        # construct ANN
        model_ann = Sequential([
            Dense(100, activation="relu"),
            Dense(100, activation="relu"),
            Dense(1, activation="linear")
        ])

        # compile model
        model_ann.compile(optimizer="adam", loss="mse", metrics=["mse"])

        # train model
        model_ann.fit(x_train, y_train, epochs=70, batch_size=20, verbose=1)

        return model_ann

    @staticmethod
    def predict_and_evaluate(model_ann, x_test, y_test):
        predictions_ann = model_ann.predict(x_test)
        mse_ann = mean_squared_error(y_test, predictions_ann)

        print(f"ANN Mean Squared Error:\t{mse_ann}") # so, it does not return any value

        return mse_ann

if __name__ == "__main__":
    result_list = [] # result list to print excel

    for i in range(100):
        result_dictionary = {}  # result dictionary to send result

        # initialize object to utilize it
        object_NN = NN()

        # set seed exactly same for reproducibility
        object_NN.set_seeds(i + 1)

        # generate matlab dataset
        matlab_dataset = object_NN.generate_matlab_dataset(1, 1000, 1750)

        # understand attached dataset
        preprocessed_dataset = object_NN.understand_dataset(matlab_dataset[0], matlab_dataset[1])

        # come up with ANN
        particular_model_ann = object_NN.construct_ann(
            preprocessed_dataset[0],
            preprocessed_dataset[2]
        )

        # predict and evaluate constructed specific model
        result_dictionary["Mean Squared Error"] = object_NN.predict_and_evaluate(particular_model_ann, preprocessed_dataset[1], preprocessed_dataset[3])
        result_list.append(result_dictionary)

    # to print Excel
    result_df = pd.DataFrame(result_list)
    result_df.to_excel(r"C:\Users\aeren\Desktop\test.xlsx", index=False)