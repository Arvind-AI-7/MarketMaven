from tensorflow.keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

class Model_LSTM:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def model(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
