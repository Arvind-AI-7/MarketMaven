from sklearn.ensemble import VotingClassifier
from MarketMaven.model.classification.knn import Model_KNN
from MarketMaven.model.classification.random_forest import Model_RandomForest

class Model_ENSEMBLE:
    def __init__(self, train_x, test_x, train_y, test_y, X_forecast):
        super(Model_ENSEMBLE, self).__init__()
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.X_forecast = X_forecast

    def model(self):
        rf = Model_RandomForest(self.train_x, self.train_y)
        rf_model = rf.model()
        knn = Model_KNN(self.train_x, self.train_y)
        knn_model = knn.model()

        estimators = [('knn', knn_model), ('rf', rf_model)]
        ensemble = VotingClassifier(estimators, voting='hard')
        ensemble.fit(self.train_x, self.train_y)
        predictions = ensemble.predict(self.test_x)

        forecast = ensemble.predict(self.X_forecast.values.reshape(1, -1))

        return predictions, forecast