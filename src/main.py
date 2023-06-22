from MarketMaven.data.data_gathering import DataGathering
from get_predictions import Get_Predictions

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    data_gathering = DataGathering(company_name='RELIANCE.NS')
    train_x, test_x, train_y, test_y, X_forecast, tomorrow = data_gathering.get_data(years=5, split=0.10) # years,train_test_split

    print("tomorrow", tomorrow)
    get_predictions = Get_Predictions(train_x, test_x, train_y, test_y, X_forecast, tomorrow)
    get_predictions.pred()

