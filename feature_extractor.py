import pandas as pd


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X_df.index = range(len(X_df))

        X_df_new = pd.concat(
            [X_df.get(['instant_t', 'windspeed', 'latitude', 'longitude',
                       'hemisphere', 'Jday_predictor', 'initial_max_wind',
                       'max_wind_change_12h', 'dist2land']),
             pd.get_dummies(X_df.nature, prefix='nature', drop_first=True)],
            # 'basin' is not used here ..but it can!
            axis=1)

        # get data from the past of the same storm (if it exists)
        past_winds = []
        past_distance = []
        for i in range(len(X_df)):
            if i - 1 < 0:
                past_winds.append(X_df['windspeed'][i])
                past_distance.append(X_df['dist2land'][i])    
            elif X_df['stormid'][i] == X_df['stormid'][i - 1]:
                past_winds.append(X_df['windspeed'][i - 1])
                past_distance.append(X_df['dist2land'][i-1])                    
            else:
                past_winds.append(X_df['windspeed'][i])
                past_distance.append(X_df['dist2land'][i]) 
        X_df_new = X_df_new.assign(
            past_windspeed_1=pd.Series(past_winds))
        X_df_new = X_df_new.assign(
            past_d2land_1=pd.Series(past_distance))
        
 
        past_winds = []
        past_distance = []
        for i in range(len(X_df_new)):
            if i - 1 < 0:
                past_winds.append(X_df_new['past_windspeed_1'][i])
                past_distance.append(X_df_new['past_d2land_1'][i])    
            elif X_df['stormid'][i] == X_df_new['stormid'][i - 1]:
                past_winds.append(X_df_new['past_windspeed_1'][i - 1])
                past_distance.append(X_df_new['past_d2land_1'][i-1])                    
            else:
                past_winds.append(X_df_new['past_windspeed_1'][i])
                past_distance.append(X_df_new['past_d2land_1'][i]) 
        X_df_new = X_df_new.assign(
            past_windspeed_2=pd.Series(past_winds))
        X_df_new = X_df_new.assign(
            past_d2land_2=pd.Series(past_distance))
        #adding max, min of variables as well as their relative locations to the center of the storm
        Xz=X_df.iloc[:,13:134];
        maxz=np.max(Xz,axis=1)
        minz=np.min(Xz,axis=1)
        loc_maxz=np.argmax(Xz,axis=1)
        loc_maxz_i=loc_maxz//11
        loc_maxz_j=loc_maxz-(loc_maxz_i*11)-1
        lox_minz=np.argmin(Xz,axis=1)
        loc_minz_i=loc_minz//11
        loc_minz_j=loc_minz-(loc_maxz_i*11)-1     
        
        
        
        XX = X_df_new.values
        return XX
