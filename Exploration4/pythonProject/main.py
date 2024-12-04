from zipfile import error

import matlab.engine
import numpy as n
import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.optimizers import Adam


class SquareRootOptimization:
    def __init__(self, max_dataset_size=10000):
        # start MATLAB engine
        self.eng = matlab.engine.start_matlab()
        self.max_dataset_size = max_dataset_size

    def generate_matlab_dataset(self, parameter_range):
        """generate square root dataset utilizing MATLAB directly from Python"""
        dataset = []
        for parameter in parameter_range:
            # direct MATLAB sqrt calculation
            sqrt_value = self.eng.sqrt(parameter)
            dataset.append([parameter, sqrt_value])

        # convert to pandas DataFrame
        data_frame = p.DataFrame(dataset, columns=["parameter", "sqrt_value"])
        return data_frame

    def train_nn_model(self, dataset, x_test_p, y_test_p):
        # Ensure dataset doesn't exceed max size
        if len(dataset) > self.max_dataset_size:
            dataset = dataset.sample(self.max_dataset_size)

        x = dataset["parameter"].values.reshape(-1, 1).astype('float32')
        y = dataset["sqrt_value"].values.astype('float32')

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Build neural network model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')  # Output layer
        ])

        # Compile and train model
        model.compile(optimizer=Adam(learning_rate=0.007), loss='mse', metrics=['mse'])
        model.fit(x_train, y_train, epochs=97, verbose=0, batch_size=16)

        # Evaluate model
        y_prediction = model.predict(x_test_p)
        mse = mean_squared_error(y_test_p, y_prediction)

        return model, mse

    def identify_high_error_regions(self, model, dataset, region_count=7, samples_per_region=10):
        # predict for entire dataset
        x = dataset["parameter"].values.reshape(-1, 1)
        y_true = dataset['sqrt_value'].values
        y_predict = model.predict(x)

        def from_2d_to_1d(matrix):
            matrix_1d = []
            for row in matrix:
                matrix_1d.extend(row)
            return matrix_1d

        y_predict = from_2d_to_1d(y_predict)

        # calculate absolute errors
        errors = y_true - y_predict

        # sort parameters by error magnitude
        error_indices = n.argsort(errors)[-region_count:]  # Top `region_count` errors
        high_error_parameters = x[error_indices].flatten()  # Flatten to ensure 1D array of scalars

        # Evenly distribute new samples across high error regions
        new_parameters = []

        for parameter in high_error_parameters:
            # Generate evenly spaced samples around the parameter
            region_samples = n.linspace(
                parameter * 0.9,
                parameter * 1.1,
                samples_per_region
            )
            new_parameters.extend(region_samples)

        print(f"NUMBERS OF NEW PARAMETERS GENERATED: {len(new_parameters)}")
        return n.array(new_parameters)

    def adaptive_optimization(self):
        # initial parameter exploration
        initial_parameters = n.linspace(1, 100, 1000)  # 100 initial samples
        initial_dataset = self.generate_matlab_dataset(initial_parameters)

        x = initial_dataset["parameter"].values.reshape(-1, 1).astype('float32')
        y = initial_dataset["sqrt_value"].values.astype('float32')

        # Train-test split
        x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(x, y, test_size=0.2, random_state=42)

        print(f"CURRENT DATASET LENGTH: {len(initial_dataset)}")

        # initial model train
        initial_model, initial_mse = self.train_nn_model(initial_dataset, x_test_p, y_test_p)
        print(f"Initial MSE = {initial_mse}")

        # iterative optimization process
        for iteration in range(9):
            # identify high error regions and generate new samples
            new_parameters = self.identify_high_error_regions(
                initial_model,
                initial_dataset,
            )

            # generate new dataset
            new_dataset = self.generate_matlab_dataset(new_parameters)

            # update combined dataset
            combined_dataset = p.concat([initial_dataset, new_dataset])

            print(f"COMBINED DATASET BEFORE TRIMMING: {len(combined_dataset)}")

            # Ensure dataset doesn't exceed max size
            if len(combined_dataset) > self.max_dataset_size:
                combined_dataset = combined_dataset.sample(self.max_dataset_size)
                print(f"COMBINED DATASET TRIMMED TO: {len(combined_dataset)}")

            print(f"CURRENT DATASET LENGTH: {len(combined_dataset)}")

            # retrain model
            new_model, new_mse = self.train_nn_model(combined_dataset, x_test_p, y_test_p)
            print(f"Iteration {iteration + 1}, MSE: {new_mse}")

            initial_dataset = combined_dataset
            initial_model = new_model
            initial_mse = new_mse

        return initial_model, initial_dataset


# Run the optimization
final_model, final_dataset = SquareRootOptimization().adaptive_optimization()