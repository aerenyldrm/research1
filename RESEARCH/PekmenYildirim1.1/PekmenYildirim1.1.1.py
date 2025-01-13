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
from tensorflow.compiler.tf2xla.python.xla import concatenate

class Curriculum:
    def __init__(self):
        # start MATLAB engine
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
        dataset_input = np.array([value[0] for value in row_form_of_dataset])
        dataset_output = np.array([value[1] for value in row_form_of_dataset])

        return dataset_input, dataset_output

    @staticmethod
    def concatenate_datasets(current_in, current_out, addition_in, addition_out):
        # concatenate current dataset with relevant approach-to-limit datasets
        dataset_in = np.concat((current_in, addition_in))
        dataset_out = np.concat((current_out, addition_out))
        return dataset_in, dataset_out

    @staticmethod
    def understand_dataset(dataset_input, dataset_output):
        # split features and target
        x = np.array([[value] for value in dataset_input])
        y = np.array([[value] for value in dataset_output])

        # split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=2)

        # copy original x_text
        copy_x_text = x_test

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

        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), np.array(copy_x_text)

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
        model_ann.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=16, batch_size=100, verbose=1)

        return model_ann

    def predict_and_evaluate(self, model_ann, x_test, y_test, copy_x_test):
        predictions_ann = model_ann.predict(x_test)
        mse_ann = mean_squared_error(y_test, predictions_ann)

        # find out high error regions
        errors = abs(predictions_ann.flatten()-y_test)
        errors_sorted = np.argsort(errors)
        maximum_errors_indices = errors_sorted[-25:] # 50 is error count to consider maximum error

        # match erroneous x_test elements with correct indices; but original x_test, not scaled x_test
        x_related_maximum_errors = [copy_x_test.flatten()[index] for index in maximum_errors_indices]

        # come up with expansion inputs and relevant output to expand current dataset
        left_approach = np.array([number - 0.5 for number in x_related_maximum_errors])
        right_approach = np.array([number + 0.5 for number in x_related_maximum_errors])
        left_approach_result = np.array([self.matlab.sqrt(number) for number in left_approach])
        right_approach_result = np.array([self.matlab.sqrt(number) for number in right_approach])
        approaches_total = np.concat((left_approach, right_approach))
        approach_results_total = np.concat((left_approach_result, right_approach_result))

        print(x_related_maximum_errors)
        print(f"ANN Mean Squared Error: {mse_ann}")

        return approaches_total, approach_results_total

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

    def curriculum(self):
        # generate initial dataset
        initial_in, initial_out = self.generate_matlab_dataset(1, 1000, 1000)
        # comprehend initial dataset
        initial_preprocessed_dataset = self.understand_dataset(initial_in, initial_out)
        fixed_x_test = initial_preprocessed_dataset[1]
        fixed_y_test = initial_preprocessed_dataset[3]
        # come up with initial ANN model
        initial_ann_model = self.construct_ann(
            initial_preprocessed_dataset[0],
            initial_preprocessed_dataset[1],
            initial_preprocessed_dataset[2],
            initial_preprocessed_dataset[3]
        )
        # predict with initial model and evaluate it
        approaches, approach_results = self.predict_and_evaluate(
            initial_ann_model,
            initial_preprocessed_dataset[1],
            initial_preprocessed_dataset[3],
            initial_preprocessed_dataset[4]
        )

        for i in range(1):
            # update and generate initial dataset iteratively
            initial_in, initial_out = self.concatenate_datasets(
                initial_in, initial_out, approaches, approach_results
            )
            # update and comprehend initial dataset iteratively
            initial_preprocessed_dataset = self.understand_dataset(initial_in, initial_out)
            # update and come up with initial ann model iteratively
            initial_ann_model = self.construct_ann(
                initial_preprocessed_dataset[0],
                initial_preprocessed_dataset[1],
                initial_preprocessed_dataset[2],
                initial_preprocessed_dataset[3]
            )
            # update and predict with initial model and evaluate it iteratively
            approaches, approach_results = self.predict_and_evaluate(
                initial_ann_model,
                fixed_x_test,
                fixed_y_test,
                initial_preprocessed_dataset[4]
            )

if __name__ == "__main__":
    # set seed for reproducibility
    Curriculum().set_seeds(11)

    # curriculum
    Curriculum().curriculum()

    # print weights of each layer
    # Curriculum().print_layer_weights(ann_model)