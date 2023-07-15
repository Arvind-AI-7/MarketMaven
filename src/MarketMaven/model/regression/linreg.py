from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class Model_LINREG:
    def __init__(self, train_x, test_x, train_y, test_y, X_forecast):
        super(Model_LINREG, self).__init__()
        self.clf = LinearRegression(n_jobs=-1)
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y.values.reshape(-1, 1)
        self.test_y = test_y.values.reshape(-1, 1)
        self.X_forecast = X_forecast
        self.sc_x = MinMaxScaler(feature_range=(0, 1))
        self.sc_y = MinMaxScaler(feature_range=(0, 1))

    def model(self):

        train_x_scaled = self.sc_x.fit_transform(self.train_x)
        test_x_scaled = self.sc_x.transform(self.test_x)

        train_y_scaled = self.sc_y.fit_transform(self.train_y)


        self.clf.fit(train_x_scaled, train_y_scaled)
        predictions_scaled = self.clf.predict(test_x_scaled)
        predictions = self.sc_y.inverse_transform(predictions_scaled)

        X_F = self.sc_x.transform(self.X_forecast.values.reshape(1, -1))

        forecast = self.clf.predict(X_F)
        forecast = self.sc_y.inverse_transform(forecast)

        return predictions, forecast

