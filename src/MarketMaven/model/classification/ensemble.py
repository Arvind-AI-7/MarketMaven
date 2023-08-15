import pickle
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

    def train(self):
        rf = Model_RandomForest(self.train_x, self.train_y)
        rf_model = rf.model()
        pickle.dump(rf_model, open("src/MarketMaven/pt_h5_pkl/rf.pickle", "wb"))

        knn = Model_KNN(self.train_x, self.train_y)
        knn_model = knn.model()
        pickle.dump(knn_model, open("src/MarketMaven/pt_h5_pkl/knn.pickle", "wb"))

        return


    def test(self):
        rf_model = pickle.load(open("src/MarketMaven/pt_h5_pkl/rf.pickle", "rb"))
        knn_model = pickle.load(open("src/MarketMaven/pt_h5_pkl/knn.pickle", "rb"))

        estimators = [('knn', knn_model), ('rf', rf_model)]
        ensemble = VotingClassifier(estimators, voting='hard')
        ensemble.fit(self.train_x, self.train_y)
        predictions = ensemble.predict(self.test_x)

        forecast = ensemble.predict(self.X_forecast.values.reshape(1, -1))

        return predictions, forecast