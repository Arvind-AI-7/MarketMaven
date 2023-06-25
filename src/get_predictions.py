from MarketMaven.model.regression.lstm import Model_LSTM
from MarketMaven.model.regression.linreg import Model_LINREG
from MarketMaven.model.classification.ensemble import Model_ENSEMBLE

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

class Get_Predictions:
    def __init__(self, train_x, test_x, train_y, test_y, X_forecast, tomorrow):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.X_forecast = X_forecast
        self.tomorrow = tomorrow

    def reg_pred(self):
        ###################### LSTM #######################
        lstm = Model_LSTM(self.train_x, self.test_x, self.train_y, self.test_y, self.X_forecast)
        lstm_pred, lstm_forecast = lstm.model()
        lstm_rmse = np.sqrt(mean_squared_error(self.test_y, lstm_pred))
        print("LSTM predictions generated")
        print("LSTM RMSE: ", lstm_rmse/(max(self.test_y)-min(self.test_y)))
        print("LSTM R2: ", round(r2_score(self.test_y, lstm_pred), 2))
        print(f"LSTM forecast for date {self.tomorrow} : {lstm_forecast}")

        ##################### LINREG ######################
        linreg = Model_LINREG(self.train_x, self.test_x, self.train_y, self.test_y, self.X_forecast)
        linreg_pred, linreg_forecast = linreg.model()
        linreg_rmse = np.sqrt(mean_squared_error(self.test_y, linreg_pred))
        print("LINREG predictions generated")
        print("LINREG RMSE: ", linreg_rmse/(max(self.test_y)-min(self.test_y)))
        print("LINREG R2: ", round(r2_score(self.test_y, linreg_pred), 2))
        print(f"LINREG forecast for date {self.tomorrow} : {linreg_forecast}")

        test_set_range = np.arange(len(self.test_y))
        plt.plot(test_set_range, self.test_y, label='Actual')
        plt.plot(test_set_range, lstm_pred, label='Predicted LSTM')
        plt.plot(test_set_range, linreg_pred, label='Predicted LINREG')
        plt.title('RELIANCE Prices Prediction')
        plt.xlabel('Date')
        plt.ylabel('Prices')
        plt.legend(loc="upper left", fontsize='small')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def clf_pred(self):
        ###################### Ensemble #######################
        ens = Model_ENSEMBLE(self.train_x, self.test_x, self.train_y, self.test_y, self.X_forecast)
        ens_pred, ens_forecast = ens.model()
        ensemble_accuracy = accuracy_score(self.test_y.values, ens_pred)
        print("Classification Accuracy: ", round(ensemble_accuracy, 2))
        print(f"Classification forecast for date {self.tomorrow} : {ens_forecast}")

