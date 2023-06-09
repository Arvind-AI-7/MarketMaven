from sklearn.preprocessing import MinMaxScaler
from MarketMaven.data.data_gathering import DataGathering
from MarketMaven.model.arima import Model_ARIMA
from MarketMaven.model.lstm import Model_LSTM

import matplotlib.pyplot as plt
from pandas.plotting import lag_plot

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    data_gathering = DataGathering(company_name='RELIANCE.NS')
    data, train_size = data_gathering.get_data()

    train_data, test_data = data[0:int(len(data)*0.8)]['Adj Close'], data[int(len(data)*0.8):]['Adj Close']

    # For Checking Cross Correlation in Data.
    # ARIMA is going to be a good model to be applied to this type of data
    # plt.figure()
    # lag_plot(data['Close'], lag=3)
    # plt.title('Reliance Stock - Autocorrelation plot with lag = 3')
    # plt.show()

    ###################### ARIMA ######################
    arima = Model_ARIMA(train_data, test_data)
    arima_pred, confidence_interval = arima.model()

    ###################### LSTM #######################
    lstm = Model_LSTM(train_data, test_data)

    test_set_range = data[int(len(data) * 0.8):].index
    plt.plot(test_set_range, arima_pred, label='Predicted ARIMA')
    plt.plot(test_set_range, test_data, label='Actual Price')
    plt.fill_between(test_set_range, confidence_interval[:,0], confidence_interval[:,1], color='gray', alpha=0.2, label='Confidence Interval')
    plt.title('RELIANCE Prices Prediction')
    plt.xlabel('Date')
    plt.ylabel('Prices')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()







