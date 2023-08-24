from MarketMaven.model.regression.cnn import Model_LSTM_CNN
from MarketMaven.model.regression.lstm_rnn import Model_LSTM_RNN
from MarketMaven.model.regression.lstm_gru_rnn import Model_LSTM_GRU
from MarketMaven.model.regression.linreg import Model_LINREG
from MarketMaven.model.classification.ensemble import Model_ENSEMBLE
from MarketMaven.model.sentiment.sentiment_analysis import Sentiment_Analysis
from MarketMaven.model.sentiment.sentiment_rnn import Sentiment_RNN
from MarketMaven.model.sentiment.sentiment_cnn import Sentiment_CNN
from utils.tickers import Tickers

import psutil
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")

class Get_Predictions:
    def __init__(self, train_x, test_x, train_y, test_y, X_forecast, tomorrow, tt_switch):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.X_forecast = X_forecast
        self.tomorrow = tomorrow
        self.tt_switch = tt_switch

    def reg_pred(self):

        ###################### LSTM_CNN #######################
        print("----------------------- CNN -----------------------")
        initial_memory = psutil.Process().memory_info().rss
        start_time = time.time()

        lstm_cnn = Model_LSTM_CNN(self.train_x, self.test_x, self.train_y, self.test_y, self.X_forecast)
        if self.tt_switch == 0:
            lstm_cnn.model()
        lstm_pred_cnn, lstm_forecast_cnn = lstm_cnn.test_cnn()
        lstm_rmse_cnn = np.sqrt(mean_squared_error(self.test_y, lstm_pred_cnn))
        lstm_mape_cnn = mean_absolute_percentage_error(self.test_y, lstm_pred_cnn)
        print("CNN predictions generated")
        print("CNN RMSE: ", lstm_rmse_cnn)
        print("CNN MAPE:", lstm_mape_cnn)
        print("CNN R2: ", round(r2_score(self.test_y, lstm_pred_cnn), 2))
        print(f"CNN forecast for date {self.tomorrow} : {lstm_forecast_cnn}")

        memory_used = 0
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_memory = psutil.Process().memory_info().rss
        memory_used = final_memory - initial_memory
        print(f"Time taken by CNN: {elapsed_time:.4f} seconds")
        print(f"Memory used by CNN: {memory_used / (1024 * 1024):.2f} MB")

        ###################### LSTM_RNN #######################
        print("--------------------- LSTM_RNN --------------------")
        initial_memory = psutil.Process().memory_info().rss
        start_time = time.time()

        lstm_rnn = Model_LSTM_RNN(self.train_x, self.test_x, self.train_y, self.test_y, self.X_forecast)
        if self.tt_switch == 0:
            lstm_rnn.model()
        lstm_pred_rnn, lstm_forecast_rnn = lstm_rnn.test_rnn()
        lstm_rmse_rnn = np.sqrt(mean_squared_error(self.test_y, lstm_pred_rnn))
        lstm_mape_rnn = mean_absolute_percentage_error(self.test_y, lstm_pred_rnn)
        print("LSTM_RNN predictions generated")
        print("LSTM_RNN RMSE: ", lstm_rmse_rnn)
        print("LSTM_RNN MAPE:", lstm_mape_rnn)
        print("LSTM_RNN R2: ", round(r2_score(self.test_y, lstm_pred_rnn), 2))
        print(f"LSTM_RNN forecast for date {self.tomorrow} : {lstm_forecast_rnn}")

        memory_used = 0
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_memory = psutil.Process().memory_info().rss
        memory_used = final_memory - initial_memory
        print(f"Time taken by LSTM_RNN: {elapsed_time:.4f} seconds")
        print(f"Memory used by LSTM_RNN: {memory_used / (1024 * 1024):.2f} MB")

        ###################### LSTM_GRU #######################
        print("------------------- LSTM_GRU_RNN ------------------")
        initial_memory = psutil.Process().memory_info().rss
        start_time = time.time()

        lstm_gru = Model_LSTM_GRU(self.train_x, self.test_x, self.train_y, self.test_y, self.X_forecast)
        if self.tt_switch == 0:
            lstm_gru.model()
        lstm_gru_pred, lstm_gru_forecast = lstm_gru.test_gru_rnn()
        lstm_gru_rmse = np.sqrt(mean_squared_error(self.test_y, lstm_gru_pred))
        lstm_gru_mape = mean_absolute_percentage_error(self.test_y, lstm_gru_pred)
        print("LSTM_GRU_RNN predictions generated")
        print("LSTM_GRU_RNN RMSE: ", lstm_gru_rmse)
        print("LSTM_GRU_RNN MAPE:", lstm_gru_mape)
        print("LSTM_GRU_RNN R2: ", round(r2_score(self.test_y, lstm_gru_pred), 2))
        print(f"LSTM_GRU_RNN forecast for date {self.tomorrow} : {lstm_gru_forecast}")

        memory_used = 0
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_memory = psutil.Process().memory_info().rss
        memory_used = final_memory - initial_memory
        print(f"Time taken by LSTM_GRU_RNN: {elapsed_time:.4f} seconds")
        print(f"Memory used by LSTM_GRU_RNN: {memory_used / (1024 * 1024):.2f} MB")

        ##################### LINREG ######################
        # linreg = Model_LINREG(self.train_x, self.test_x, self.train_y, self.test_y, self.X_forecast)
        # linreg_pred, linreg_forecast = linreg.model()
        # linreg_rmse = np.sqrt(mean_squared_error(self.test_y, linreg_pred))
        # print("LINREG predictions generated")
        # print("LINREG RMSE: ", linreg_rmse)
        # print("LINREG R2: ", round(r2_score(self.test_y, linreg_pred), 2))
        # print(f"LINREG forecast for date {self.tomorrow} : {linreg_forecast}")

        test_set_range = np.arange(len(self.test_y))
        plt.plot(test_set_range, self.test_y, label='Actual')
        plt.plot(test_set_range, lstm_pred_cnn, label='Predicted CNN')
        plt.plot(test_set_range, lstm_pred_rnn, label='Predicted LSTM_RNN')
        plt.plot(test_set_range, lstm_gru_pred, label='Predicted LSTM_GRU_RNN')
        # plt.plot(test_set_range, linreg_pred, label='Predicted LINREG')
        plt.title('AAPL Prices Prediction')
        plt.xlabel('Days')
        plt.ylabel('Prices')
        plt.legend(loc="upper left", fontsize='small')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # plt.show()
        plt.savefig('src/MarketMaven/screenshots/regression.png')

    def clf_pred(self):
        ###################### Ensemble #######################
        ens = Model_ENSEMBLE(self.train_x, self.test_x, self.train_y, self.test_y, self.X_forecast)
        if self.tt_switch == 0:
            ens.train()
        ens_pred, ens_forecast = ens.test()
        ensemble_accuracy = accuracy_score(self.test_y.values, ens_pred)
        print("Classification Accuracy: ", round(ensemble_accuracy, 2))
        print(f"Classification forecast for date {self.tomorrow} : {ens_forecast}")


    def sent_pred(self):

        tk = Tickers()

        ###################### Sentiment Analysis #######################
        # sa = Sentiment_Analysis(tk.tickers, tk.tickers_sector, tk.tickers_dict, tk.tickers_name, tk.number_of_shares, tk.source_url)
        # sa.sentiment()

        ######################## Sentiment RNN ##########################
        print("--------------------- SENT_RNN --------------------")
        initial_memory = psutil.Process().memory_info().rss
        start_time = time.time()

        sr = Sentiment_RNN(tk.tickers, tk.tickers_sector, tk.tickers_dict, tk.tickers_name, tk.number_of_shares, tk.source_url)
        if self.tt_switch == 0:
            sr.train_sent_rnn()
        sr.test_sent_rnn()

        memory_used = 0
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_memory = psutil.Process().memory_info().rss
        memory_used = final_memory - initial_memory
        print(f"Time taken by Sent_RNN: {elapsed_time:.4f} seconds")
        print(f"Memory used by Sent_RNN: {memory_used / (1024 * 1024):.2f} MB")

        ######################## Sentiment CNN ##########################
        print("--------------------- SENT_CNN --------------------")
        initial_memory = psutil.Process().memory_info().rss
        start_time = time.time()

        sc = Sentiment_CNN(tk.tickers, tk.tickers_sector, tk.tickers_dict, tk.tickers_name, tk.number_of_shares, tk.source_url)
        if self.tt_switch == 0:
            sc.train_sent_cnn()
        sc.test_sent_cnn()

        memory_used = 0
        end_time = time.time()
        elapsed_time = end_time - start_time
        final_memory = psutil.Process().memory_info().rss
        memory_used = final_memory - initial_memory
        print(f"Time taken by Sent_CNN: {elapsed_time:.4f} seconds")
        print(f"Memory used by Sent_CNN: {memory_used / (1024 * 1024):.2f} MB")
