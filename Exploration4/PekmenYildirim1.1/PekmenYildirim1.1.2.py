import math
import random as rd
import numpy as np
import tensorflow as tf
import keras as kr
import matlab.engine as me
from keras import Sequential
from keras.api.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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

    def generate_constant_test_dataset(self, start, end, step):
        # produce test parameters send these to MATLAB for calculation
        parameter_set = [float(x) for x in list(np.linspace(start, end, step))]

        # generate test dataset with list comprehension
        row_form_of_dataset = [[parameter, self.matlab.sqrt(parameter)] for parameter in parameter_set]

        x_test_entire = np.array([value[0] for value in row_form_of_dataset])
        y_test_entire = np.array([value[1] for value in row_form_of_dataset])

        return x_test_entire, y_test_entire

    def generate_matlab_dataset(self, start, end, step):
        # produce initial parameters send these to MATLAB for calculations
        parameter_set = [float(x) for x in list(np.linspace(start, end, step))]

        # generate initial dataset with list comprehension
        row_form_of_dataset = [[parameter, self.matlab.sqrt(parameter)] for parameter in parameter_set]
        dataset_input = np.array([value[0] for value in row_form_of_dataset])
        dataset_output = np.array([value[1] for value in row_form_of_dataset])

        # shuffle data to prevent incorrect patterns
        indices_shuffle = np.random.permutation(len(dataset_input))
        dataset_input = dataset_input[indices_shuffle]
        dataset_output = dataset_output[indices_shuffle]

        return dataset_input, dataset_output

    @staticmethod
    def concatenate_datasets(current_in, current_out, addition_in, addition_out):
        # concatenate current dataset with relevant approach-to-limit datasets
        dataset_in = np.concat((current_in, addition_in))
        dataset_out = np.concat((current_out, addition_out))
        return dataset_in, dataset_out

    @staticmethod
    def understand_dataset(dataset_in, dataset_out, x_test_entire, y_test_entire):
        """Utilize entire dataset for train"""
        # split features and target
        x = np.array(dataset_in).reshape(-1, 1)
        y = np.array(dataset_out).reshape(-1, 1) # 2d for neural network compatibility
        x_test_entire = np.array(x_test_entire).reshape(-1, 1)
        y_test_entire = np.array(y_test_entire).reshape(-1, 1)

        # normalize or standardize x data to prevent instability during training
        scaler_x = StandardScaler()
        x_train_entire = scaler_x.fit_transform(x)
        x_test_entire = scaler_x.transform(x_test_entire)

        # optionally scale outputs, y, only if it is necessary; comment out following if unnecessary
        scaler_y = StandardScaler()
        y_train_entire = scaler_y.fit_transform(y)
        y_test_entire = scaler_y.transform(y_test_entire)

        return x_train_entire, x_test_entire, y_train_entire, y_test_entire

    @staticmethod
    def construct_ann(x_train_entire, y_train_entire, epoch_number):
        # construct ANN
        model_ann = Sequential([
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(1, activation="linear") # output
        ])

        # compile model
        model_ann.compile(optimizer="adam", loss="mse", metrics=["mse"])

        # train model
        model_ann.fit(x_train_entire, y_train_entire, epochs=epoch_number, batch_size=20, verbose=1)

        return model_ann

    def predict_and_evaluate(self, model_ann, x_test_entire, y_test_entire, copy_x_test_entire, iteration):
        predictions_ann = model_ann.predict(x_test_entire)
        mse_ann = mean_squared_error(y_test_entire, predictions_ann)

        # find out high error regions
        errors = abs(predictions_ann.flatten()-y_test_entire.flatten())
        errors_sorted = np.argsort(errors)
        maximum_errors_indices = errors_sorted[-100:] # 100 is error count to consider maximum error

        # match erroneous x_test elements with correct indices; but original x_test, not scaled x_test
        x_related_maximum_errors = [copy_x_test_entire[index] for index in maximum_errors_indices]

        # come up with expansion inputs and relevant output to expand current dataset
        left_approach = np.array([number - iteration for number in x_related_maximum_errors])
        right_approach = np.array([number + iteration for number in x_related_maximum_errors])
        left_approach_result = np.array([self.matlab.sqrt(number) for number in left_approach])
        right_approach_result = np.array([self.matlab.sqrt(number) for number in right_approach])
        approaches_total = np.concat((left_approach, right_approach))
        approach_results_total = np.concat((left_approach_result, right_approach_result))

        print(x_related_maximum_errors)
        print(f"ANN Mean Squared Error: {mse_ann}")

        return approaches_total, approach_results_total

    def curriculum(self):
        # generate constant test dataset
        constant_test_dataset_in, constant_test_dataset_out = self.generate_constant_test_dataset(1, 1000, 10000)
        # generate initial dataset
        initial_in, initial_out = self.generate_matlab_dataset(1, 1000, 1000)
        # comprehend initial dataset
        initial_preprocessed_dataset = self.understand_dataset(initial_in, initial_out, constant_test_dataset_in, constant_test_dataset_out)
        # come up with initial ANN model
        initial_ann_model = self.construct_ann(
            initial_preprocessed_dataset[0],
            initial_preprocessed_dataset[2],
            10
        )
        # predict with initial model and evaluate it
        approaches, approach_results = self.predict_and_evaluate(
            initial_ann_model,
            initial_preprocessed_dataset[1],
            initial_preprocessed_dataset[3],
            constant_test_dataset_in,
            math.pow(2, - 1)
        )

        for i in range(3):
            # update and generate initial dataset iteratively
            initial_in, initial_out = self.concatenate_datasets(
                initial_in, initial_out, approaches, approach_results
            )
            # update and comprehend initial dataset iteratively
            initial_preprocessed_dataset = self.understand_dataset(initial_in, initial_out, constant_test_dataset_in, constant_test_dataset_out)
            # update and come up with initial ann model iteratively
            initial_ann_model = self.construct_ann(
                initial_preprocessed_dataset[0],
                initial_preprocessed_dataset[2],
                10 + int((10 * (i+1) * 200) / 1000)
            )
            # update and predict with initial model and evaluate it iteratively
            approaches, approach_results = self.predict_and_evaluate(
                initial_ann_model,
                initial_preprocessed_dataset[1],
                initial_preprocessed_dataset[3],
                constant_test_dataset_in,
                math.pow(i + 3, - 1)
            )

if __name__ == "__main__":
    # set seed for reproducibility
    Curriculum().set_seeds(2)

    # curriculum
    Curriculum().curriculum()