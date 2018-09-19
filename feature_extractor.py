import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        X_df_new = X_df_new.fillna(-1)
        X_df = X_df.fillna(-1)
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
            elif X_df['stormid'][i] == X_df['stormid'][i - 1]:
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
        Xz=X_df.iloc[:,X_df.columns.get_loc("z_0_0")+1:X_df.columns.get_loc("z_10_10")+1]
        maxz=np.max(Xz,axis=1)
        minz=np.min(Xz,axis=1)
        loc_maxz=np.argmax(Xz.values,axis=1)
        loc_maxz_i=loc_maxz//11
        loc_maxz_j=loc_maxz-(loc_maxz_i*11)-1
        loc_minz=np.argmin(Xz.values,axis=1)
        loc_minz_i=loc_minz//11
        loc_minz_j=loc_minz-(loc_maxz_i*11)-1     
        X_df_new = X_df_new.assign(
            maxz=pd.Series(maxz),minz=pd.Series(minz),loc_maxz_i=pd.Series(loc_maxz_i),loc_maxz_j=pd.Series(loc_maxz_j),
            loc_minz_i=pd.Series(loc_minz_i),loc_minz_j=pd.Series(loc_minz_j))
        
        
                
        Xslp=X_df.iloc[:,X_df.columns.get_loc("slp_0_0")+1:X_df.columns.get_loc("slp_10_10")+1]
        maxslp=np.max(Xslp,axis=1)
        minslp=np.min(Xslp,axis=1)
        loc_maxslp=np.argmax(Xslp.values,axis=1)
        loc_maxslp_i=loc_maxslp//11
        loc_maxslp_j=loc_maxslp-(loc_maxslp_i*11)-1
        loc_minslp=np.argmin(Xslp.values,axis=1)
        loc_minslp_i=loc_minslp//11
        loc_minslp_j=loc_minslp-(loc_maxslp_i*11)-1   
        X_df_new = X_df_new.assign(
                maxslp=pd.Series(maxslp),minslp=pd.Series(minslp),loc_maxslp_i=pd.Series(loc_maxslp_i),loc_maxslp_j=pd.Series(loc_maxslp_j),
                loc_minslp_i=pd.Series(loc_minslp_i),loc_minslp_j=pd.Series(loc_minslp_j))
        
        Xhum=X_df.iloc[:,X_df.columns.get_loc("hum_0_0")+1:X_df.columns.get_loc("hum_10_10")+1];
        maxhum=np.max(Xhum,axis=1)
        minhum=np.min(Xhum,axis=1)
        loc_maxhum=np.argmax(Xhum.values,axis=1)
        loc_maxhum_i=loc_maxhum//11
        loc_maxhum_j=loc_maxhum-(loc_maxhum_i*11)-1
        loc_minhum=np.argmin(Xhum.values,axis=1)
        loc_minhum_i=loc_minhum//11
        loc_minhum_j=loc_minhum-(loc_maxhum_i*11)-1   
        X_df_new = X_df_new.assign(
                maxhum=pd.Series(maxhum),minhum=pd.Series(minhum),loc_maxhum_i=pd.Series(loc_maxhum_i),loc_maxhum_j=pd.Series(loc_maxhum_j),
                loc_minhum_i=pd.Series(loc_minhum_i),loc_minhum_j=pd.Series(loc_minhum_j))        
        
        
        Xsst=X_df.iloc[:,X_df.columns.get_loc("sst_0_0")+1:X_df.columns.get_loc("sst_10_10")+1];
        maxsst=np.max(Xsst,axis=1)
        minsst=np.min(Xsst,axis=1)
        loc_maxsst=np.argmax(Xsst.values,axis=1)
        loc_maxsst_i=loc_maxsst//11
        loc_maxsst_j=loc_maxsst-(loc_maxsst_i*11)-1
        loc_minsst=np.argmin(Xsst.values,axis=1)
        loc_minsst_i=loc_minsst//11
        loc_minsst_j=loc_minsst-(loc_maxsst_i*11)-1 
        X_df_new = X_df_new.assign(
                maxsst=pd.Series(maxsst),minsst=pd.Series(minsst),loc_maxsst_i=pd.Series(loc_maxsst_i),loc_maxsst_j=pd.Series(loc_maxsst_j),
                loc_minsst_i=pd.Series(loc_minsst_i),loc_minsst_j=pd.Series(loc_minsst_j))
        
        Xvo700=X_df.iloc[:,X_df.columns.get_loc("vo700_0_0")+1:X_df.columns.get_loc("vo700_10_10")+1];
        maxvo700=np.max(Xvo700,axis=1)
        minvo700=np.min(Xvo700,axis=1)
        loc_maxvo700=np.argmax(Xvo700.values,axis=1)
        loc_maxvo700_i=loc_maxvo700//11
        loc_maxvo700_j=loc_maxvo700-(loc_maxvo700_i*11)-1
        loc_minvo700=np.argmin(Xvo700.values,axis=1)
        loc_minvo700_i=loc_minvo700//11
        loc_minvo700_j=loc_minvo700-(loc_maxvo700_i*11)-1   
        X_df_new = X_df_new.assign(
                maxvo700=pd.Series(maxvo700),minvo700=pd.Series(minvo700),loc_maxvo700_i=pd.Series(loc_maxvo700_i),loc_maxvo700_j=pd.Series(loc_maxvo700_j),
                loc_minvo700_i=pd.Series(loc_minvo700_i),loc_minvo700_j=pd.Series(loc_minvo700_j))
        
        
        XX = X_df_new.values
        scaler=StandardScaler()
        scaler.fit(XX)
        XX = X_df_new.values
        return XX
