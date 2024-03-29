from MarketMaven.data.clf_data_gathering import CLFDataGathering
from MarketMaven.data.reg_data_gathering import REGDataGathering

from get_predictions import Get_Predictions

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    tt_switch = 0  # 0 for train / 1 for test

    # Regression
    print("******************************* Regression *******************************")
    data_gathering = REGDataGathering(company_name='AAPL')
    train_x, test_x, train_y, test_y, X_forecast, tomorrow = data_gathering.get_data(years=5, split=0.10) # years,train_test_split
    get_predictions = Get_Predictions(train_x, test_x, train_y, test_y, X_forecast, tomorrow, tt_switch)
    get_predictions.reg_pred()

    # Classification
    print("***************************** Classification *****************************")
    data_gathering = CLFDataGathering(company_name='AAPL')
    train_x, test_x, train_y, test_y, X_forecast, tomorrow = data_gathering.get_data(years=5, split=0.10)  # years,train_test_split
    get_predictions = Get_Predictions(train_x, test_x, train_y, test_y, X_forecast, tomorrow, tt_switch)
    get_predictions.clf_pred()


    # Sentiment Analysis
    print("*************************** Sentiment Analysis ***************************")
    get_predictions = Get_Predictions(train_x, test_x, train_y, test_y, X_forecast, tomorrow, tt_switch)
    get_predictions.sent_pred()