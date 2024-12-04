import matlab.engine
import numpy as n
import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.optimizers import Adam

class SquareRootOptimization:
    def __init__(self):
        # start MATLAB engine
        self.eng = matlab.engine.start_matlab()

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

    def train_nn_model(self, dataset):
        x = dataset["parameter"].values.reshape(-1, 1).astype('float32')
        y = dataset["sqrt_value"].values.astype('float32')

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Build neural network model
        model = Sequential([
            Dense(16, activation='relu', input_dim=1),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # Output layer
        ])

        # Compile and train model
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mse'])
        model.fit(x_train, y_train, epochs=53   , verbose=0, batch_size=16)

        # Evaluate model
        y_prediction = model.predict(x_test)
        mse = mean_squared_error(y_test, y_prediction)

        return model, mse

    def identify_high_error_regions(self, model, dataset, region_count=2, sample_count_per_region=7):
        # predict for entire dataset
        x = dataset["parameter"].values.reshape(-1, 1)
        y_true = dataset['sqrt_value'].values
        y_predict = model.predict(x)

        # calculate absolute errors
        errors = n.abs(y_true - y_predict)

        # sort parameters by error magnitude
        error_indices = n.argsort(errors)[-region_count:]  # Top `region_count` errors
        high_error_parameters = x[error_indices].flatten()  # Flatten to ensure 1D array of scalars

        # generate new samples around these high error points
        new_parameters = []
        for parameter in high_error_parameters:
            # parameter is now a scalar
            new_parameters.extend(n.random.normal(parameter, parameter * 0.1, sample_count_per_region))

        return n.array(new_parameters)

    def adaptive_optimization(self):
        # initial parameter exploration
        initial_parameters = n.linspace(1, 100, 10)
        initial_dataset = self.generate_matlab_dataset(initial_parameters)

        # initial model train
        initial_model, initial_mse = self.train_nn_model(initial_dataset)
        print(f"Initial MSE = {initial_mse}")

        # iterative optimization process
        for iteration in range(3):
            # identify high error regions and generate new samples
            new_parameters = self.identify_high_error_regions(initial_model, initial_dataset)

            # generate new dataset
            new_dataset = self.generate_matlab_dataset(new_parameters)

            # update combined dataset
            combined_dataset = p.concat([initial_dataset, new_dataset]).sample(frac=1).reset_index(drop=True)

            # retrain model
            new_model, new_mse = self.train_nn_model(combined_dataset)
            print(f"Iteration {iteration + 1}, MSE: {new_mse}")

            initial_dataset = combined_dataset
            initial_model = new_model
            initial_mse = new_mse

        return initial_model, initial_dataset

final_model, final_dataset = SquareRootOptimization().adaptive_optimization()