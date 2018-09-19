import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix

class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        # you don't have to do anything here
        # - unless you want to use a combined
        # feature extractor/regressor (like deep)
        pass

    def transform(self, X_df):
        X_df_new = pd.concat(
            [X_df.get(['windspeed', 'latitude', 'longitude',
                       'hemisphere', 'Jday_predictor', 'initial_max_wind',
                       'max_wind_change_12h', 'dist2land','basin']),
             pd.get_dummies(X_df.nature, prefix='nature', drop_first=True)],
            # 'basin' is not used here ..but it can!
            axis=1)
        X_df_new = X_df_new.fillna(-1)
        XX = X_df_new.values
        scaler=StandardScaler()
        scaler.fit(XX)
        return XX
