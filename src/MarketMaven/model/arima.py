import pmdarima as pm

class Model_ARIMA:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def model(self):
        model = pm.auto_arima(self.train_data, seasonal=True, m=1)
        model_pred, confidence_interval = model.predict(self.test_data.shape[0], return_conf_int=True, alpha=0.05)
        return model_pred, confidence_interval

