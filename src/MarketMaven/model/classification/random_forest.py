from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Model_RandomForest:
    def __init__(self, train_x, train_y):
        super(Model_RandomForest, self).__init__()
        self.rf = RandomForestClassifier()
        self.params_rf = {'n_estimators': [10,20,30,40,50]}
        self.train_x = train_x
        self.train_y = train_y

    def model(self):
        rf_gs = GridSearchCV(self.rf, self.params_rf, cv=5)
        rf_gs.fit(self.train_x, self.train_y)
        rf_best = rf_gs.best_estimator_

        return rf_best