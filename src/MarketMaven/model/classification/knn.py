from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
np.random.seed(15)

class Model_KNN:
    def __init__(self, train_x, train_y):
        super(Model_KNN, self).__init__()
        self.knn = KNeighborsClassifier()
        self.params_knn = {'n_neighbors': np.arange(1,20)}
        self.train_x = train_x
        self.train_y = train_y

    def model(self):
        knn_gs = GridSearchCV(self.knn, self.params_knn, cv=5)
        knn_gs.fit(self.train_x, self.train_y)
        knn_best = knn_gs.best_estimator_

        return knn_best