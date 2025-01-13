import math
import numpy as np
import keras as kr
import random as rd
import pandas as pd
import tensorflow as tf
import matlab.engine as me
from keras import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class Curriculum:
    def __init__(
        self,
        seed,
        ref_dataset,
        target_range,
        iteration,
        initial_error_region_count,
        initial_epoch,
        initial_batch_size,
        initial_learning_rate
    ):
        self.seed = seed
        self.ref_dataset = ref_dataset
        self.target_range = target_range
        self.iteration = iteration
        self.initial_error_region_count = initial_error_region_count
        self.initial_epoch = initial_epoch
        self.initial_batch_size = initial_batch_size
        self.initial_learning_rate = initial_learning_rate
        self.matlab = me.start_matlab() # start MATLAB engine

    def set_seeds(self):
        # set seeds for reproducibility
        rd.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # for tensorflow and keras
        kr.api.utils.set_random_seed(self.seed)
        kr.utils.set_random_seed(self.seed)

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
        dataset_in = np.array([value[0] for value in row_form_of_dataset])
        dataset_out = np.array([value[1] for value in row_form_of_dataset])

        # shuffle data to prevent incorrect patterns
        indices_shuffle = np.random.permutation(len(dataset_in))
        dataset_in = dataset_in[indices_shuffle]
        dataset_out = dataset_out[indices_shuffle]

        return dataset_in, dataset_out

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
    def construct_ann(x_train_entire, y_train_entire, epoch_value, learning_rate_value, batch_size_value):
        # construct ANN
        model_ann = Sequential([
            Dense(100, activation="relu"),
            Dense(100, activation="relu"),
            Dense(1, activation="linear") # output
        ])

        # compile model
        model_ann.compile(optimizer=Adam(learning_rate=learning_rate_value), loss="mse", metrics=["mse"])

        # train model
        model_ann.fit(x_train_entire, y_train_entire, epochs=epoch_value, batch_size=batch_size_value, verbose=1)

        print(f"Learning Rate Value: {learning_rate_value}")

        return model_ann

    def predict_and_evaluate(self, model_ann, x_test_entire, y_test_entire, copy_x_test_entire, iteration):
        predictions_ann = model_ann.predict(x_test_entire)
        mse_ann = mean_squared_error(y_test_entire, predictions_ann)

        # find out high error regions
        errors = abs(predictions_ann.flatten()-y_test_entire.flatten())
        errors_sorted = np.argsort(errors)
        maximum_errors_indices = errors_sorted[- self.initial_error_region_count:] # error count to consider maximum error

        # match erroneous x_test elements with correct indices; but original x_test, not scaled x_test
        x_related_maximum_errors = [copy_x_test_entire[index] for index in maximum_errors_indices]

        # come up with expansion inputs and relevant output to expand current dataset
        left_approach = np.array([number - iteration for number in x_related_maximum_errors])
        right_approach = np.array([number + iteration for number in x_related_maximum_errors])
        left_approach_result = np.array([self.matlab.sqrt(number) for number in left_approach])
        right_approach_result = np.array([self.matlab.sqrt(number) for number in right_approach])
        approaches_total = np.concat((left_approach, right_approach))
        approach_results_total = np.concat((left_approach_result, right_approach_result))

        print(f"ANN Mean Squared Error: {mse_ann}")

        return approaches_total, approach_results_total, mse_ann

    def curriculum(self):
        self.set_seeds()

        is_successful = True # to keep whether approach is successful or not

        mse_per_seed = {}
        # generate constant test dataset
        constant_test_dataset_in, constant_test_dataset_out = self.generate_constant_test_dataset(
            self.target_range[0], self.target_range[1], self.target_range[2]
        )
        # generate initial dataset
        initial_in, initial_out = self.generate_matlab_dataset(
            self.ref_dataset[0],
            self.ref_dataset[1],
            self.ref_dataset[2]
        )
        # comprehend initial dataset
        initial_preprocessed_dataset = self.understand_dataset(initial_in, initial_out, constant_test_dataset_in, constant_test_dataset_out)
        # come up with initial ANN model
        initial_ann_model = self.construct_ann(
            initial_preprocessed_dataset[0],
            initial_preprocessed_dataset[2],
            self.initial_epoch,
            self.initial_learning_rate,
            self.initial_batch_size
        )
        # predict with initial model and evaluate it
        approaches, approach_results, mse = self.predict_and_evaluate(
            initial_ann_model,
            initial_preprocessed_dataset[1],
            initial_preprocessed_dataset[3],
            constant_test_dataset_in,
            math.pow(2, - 1)
        )

        mse_per_seed["Iteration 1"] = mse

        current_minimum_mse = mse # minimum mse to keep track minimum to check whether result is consistent or not

        for j in range(self.iteration):
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
                self.initial_epoch + int((self.initial_epoch * (j + 1) * 2 * self.initial_error_region_count) / self.ref_dataset[1]),
                self.initial_learning_rate / math.sqrt((self.ref_dataset[1] + (j + 1) * 2 * self.initial_error_region_count) / self.ref_dataset[1]),
                self.initial_batch_size + int(math.sqrt((self.ref_dataset[1] + (j + 1) * 2 * self.initial_error_region_count) / self.ref_dataset[1]))
            )
            # update and predict with initial model and evaluate it iteratively
            approaches, approach_results, mse = self.predict_and_evaluate(
                initial_ann_model,
                initial_preprocessed_dataset[1],
                initial_preprocessed_dataset[3],
                constant_test_dataset_in,
                math.pow(j + 3, - 1)
            )

            if mse <= current_minimum_mse: current_minimum_mse = mse
            else: is_successful = False

            mse_per_seed[f"Iteration {j + 2}"] = mse

        mse_per_seed["Success"] = is_successful

        return mse_per_seed, is_successful

if __name__ == "__main__":
    result = [] # result to store Excel export materials
    successful_count = 0
    total_count = 0

    for i in range(100):
        # generate curriculum object
        curriculum = Curriculum(
            i + 1,
            (1, 1000, 1000),
            (1, 1000, 10000),
            2,
            100,
            50,
            20,
            0.001
        )

        # set seed for reproducibility and curriculum and construct result
        curriculum_return = curriculum.curriculum()
        result.append(curriculum_return[0])

        # keep track successful count and total count
        if curriculum_return[1] is True: successful_count += 1
        total_count += 1

    result.append({"Success Ratio": f"{successful_count} / {total_count}"})

    # result to Excel
    result_df = pd.DataFrame(result)
    result_df.to_excel(r"C:\Users\aeren\Desktop\test.xlsx", index=False)