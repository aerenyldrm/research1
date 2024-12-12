import random as rd
import numpy as np
import tensorflow as tf
import keras as kr
import matlab.engine as me
from keras import Sequential
from keras.api.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class Curriculum:
    def __init__(self):
        # start MATLAB engine
        self.matlab = me.start_matlab()

    def generate_matlab_dataset(self, start, end, step):
        # produce initial parameters send these to MATLAB for calculations
        parameter_set = [float(x) for x in list(np.linspace(start, end, step))]

        # generate initial dataset with list comprehension
        return [[parameter, self.matlab.sqrt(parameter)] for parameter in parameter_set]

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

    @staticmethod
    def understand_dataset(dataset):
        # split features and target
        x = [[value_calculation[0]] for value_calculation in dataset]
        y = [[value_calculation[1]] for value_calculation in dataset]

        # split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

        # normalize or standardize data to prevent instability during training
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        # x_train, x_test, y_train, y_test as 2d arrays with proper scale
        x_train = scaler_x.fit_transform(x_train)
        x_test = scaler_x.transform(x_test)
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)

        # convert back y_train and y_test back to 1d lists
        y_train = [item[0] for item in y_train]
        y_test = [item[0] for item in y_test]

        x_train = [[value[0]] for value in x_train]
        x_test = [[value[0]] for value in x_test]

        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    @staticmethod
    def construct_ann(x_train, x_test, y_train, y_test):
        # construct ANN
        model_ann = Sequential([
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(1, activation="linear") # output
        ])

        # compile model
        model_ann.compile(optimizer="adam", loss="mse", metrics=["mse"])

        # train model
        model_ann.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=128, batch_size=100, verbose=1)

        return model_ann

    @staticmethod
    def predict_and_evaluate(model_ann, x_test, y_test):
        predictions_ann = model_ann.predict(x_test)
        mse_ann = mean_squared_error(y_test, predictions_ann)

        print(f"ANN Mean Squared Error: {mse_ann}")

    @staticmethod
    def print_layer_weights(model_ann):
        print("\n--- Neural Network Weights ---\n")
        for i, layer in enumerate(model_ann.layers):
            weights = layer.get_weights() # retrieve weights and biases
            if weights: # check if layer carries weights
                weight_matrix, bias_vector = weights
                print(f"Layer {i + 1} - {layer.name}")
                print(f"Weights:\n{weight_matrix}")
                print(f"Biases:\n{bias_vector}\n")
            else:
                print(f"Layer {i + 1} - {layer.name} carries no weights.\n")

if __name__ == "__main__":
    # set seed for reproducibility
    Curriculum().set_seeds(2)

    # generate initial dataset
    initial_dataset = Curriculum().generate_matlab_dataset(1, 1000, 1000)

    # comprehend dataset
    preprocessed_dataset = Curriculum().understand_dataset(initial_dataset)

    # come up with ANN model
    ann_model = Curriculum().construct_ann(
        preprocessed_dataset[0],
        preprocessed_dataset[1],
        preprocessed_dataset[2],
        preprocessed_dataset[3]
    )

    # predict and evaluate
    Curriculum().predict_and_evaluate(ann_model, preprocessed_dataset[1], preprocessed_dataset[3])

    # print weights of each layer
    Curriculum().print_layer_weights(ann_model)