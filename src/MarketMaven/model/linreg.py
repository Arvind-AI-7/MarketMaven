from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np


class Model_LINREG:
    def __init__(self, train_x, test_x, train_y, test_y, X_forecast):
        super(Model_LINREG, self).__init__()
        self.clf = LinearRegression(n_jobs=-1)
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y.values.reshape(-1, 1)
        self.test_y = test_y.values.reshape(-1, 1)
        self.X_forecast = X_forecast

    def model(self):
        self.clf.fit(self.train_x, self.train_y)
        predictions = self.clf.predict(self.test_x)

        forecast = self.clf.predict(self.X_forecast.values.reshape(1, -1))

        return predictions, forecast

