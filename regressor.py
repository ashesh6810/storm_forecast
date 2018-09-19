from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestRegressor
class Regressor(BaseEstimator):
    def __init__(self):
         self.reg = RandomForestRegressor(max_features=0.2,n_estimators=400, random_state=0,oob_score = True,max_depth=20)
        
    def fit(self, X, y):
        y = y - X[:,1]
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X) + X[:,1]
