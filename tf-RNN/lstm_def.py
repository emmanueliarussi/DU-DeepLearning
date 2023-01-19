
# coding: utf-8

# In[247]:


import re
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sklearn
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import r2_score
import time

from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, LSTM, Bidirectional, RepeatVector, MaxPooling1D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import TimeseriesGenerator


class anom_lstm():
    """
    
    """
    def __init__(self, train, windowL = 30, cat_num = 5, Verbose=True, oldPred=True):
        self.train_orig = train #raw training and validation data
        self.cat_num = cat_num # num unique values necesasry not to be categorical
        self.windowL = windowL # window length to use for one-ahead predictions
        self.quality_sensors = [x for x in train.columns if x !='time']
        self.train_loss = []
        self.val_loss = []
        self.train_start = train.time.iloc[0]
#         self.year_start = year_start
        self.Verbose = Verbose
        self.oldPred = oldPred # Tells if a new prediction is needed (eg. after new data or more training)

    def process_train(self, train_ratio = 0.9, train_val_ratio = 0.1, bats = 256):
        self.batch_size = bats
        self.train_ratio = train_ratio
        self.train_val_ratio = train_val_ratio
        self.train_n = int(len(self.train_orig)*train_ratio)
        self.val_n = len(self.train_orig) - self.train_n
        n = len(self.train_orig)
        self.train_raw = self.train_orig.iloc[:int(n*train_ratio)]
        self.validate_raw = self.train_orig.iloc[int(n*train_ratio):]
        
        self.train_pred = False # Whether or not predictions have been made on the training set
        # Eliminate constant sensors
        self.constant = [x for x in self.train_raw.columns if self.train_raw[x].nunique() == 1]
        self.train_raw.drop(self.constant, axis=1, inplace=True)
        self.ids = list(self.train_raw.columns)
        
        # Find any sensor highly correlated with time.
        self.timeSensors = []
        timeser = pd.Series(self.train_raw[['time']].values.reshape(-1))
        for sensor in self.ids:
            sensorSeries = pd.Series(self.train_raw[sensor].values.reshape(-1))
            if np.abs(timeser.corr(sensorSeries)) >= 0.9:
                self.timeSensors.append(sensor)
        
        # Difference the sensors highly correlated with time
        timedf = pd.DataFrame()
        timedf['time'] = self.train_raw.time
        for sensor in self.timeSensors:
            if sensor == 'time':
                continue
            parts = re.split(r'\_\_',sensor) #THIS IS FOR JCSAT NAMING CONVENTIONS
#             parts = re.split(r':',sensor) #THIS IS FOR AM10 NAMING CONVENTIONS
            diffname = parts[0]+'_DIFF__'+parts[1]
            timedf[diffname]=self.train_raw[sensor].diff()
        timedf.fillna(method='bfill', inplace=True)

        # Checking for difference still correlated with time
        ids = list(timedf.columns)
        self.difftimeSensors = []
        timeser = pd.Series(timedf[['time']].values.reshape(-1))
        for sensor in ids:
            sensorSeries = pd.Series(timedf[sensor].values.reshape(-1))
            if np.abs(timeser.corr(sensorSeries)) >= 0.9:
                self.difftimeSensors.append(sensor)
        # Drop diff'ed sensors correlated with time (and drop time)
        timedf.drop(self.difftimeSensors, axis=1, inplace=True)
        
        # Make a scaler that one-hot encodes categorical sensors and scales the others. This should
        # probably be fit on the Orig dataframes with 'time' dropped.
        unique_vals = self.train_raw.nunique()
        self.cat_sensors = [x for x in list(set(self.ids)-set(self.timeSensors)) if unique_vals[x] <= self.cat_num]
        self.num_sensors = [x for x in list(set(self.ids)-set(self.timeSensors)) if unique_vals[x] > self.cat_num]

        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.num_sensors),
                ('cat', self.categorical_transformer, self.cat_sensors)])
                                            
        frames = [self.train_raw[list(set(self.ids)-set(self.timeSensors))], timedf]
        self.fittingdf = pd.concat(frames, axis=1)
        self.preprocessor.fit(self.fittingdf)
        trans = self.preprocessor.transform(self.fittingdf) #transformed training data
        try:
            self.train_transformed = pd.DataFrame(trans.todense())
        except:
            self.train_transformed = pd.DataFrame(trans)
        self.train_transformed['day_time_x'] = np.cos(self.train_raw['time']%(24*60*60*1000)*(2*np.pi/(24*60*60*1000)))
        self.train_transformed['day_time_y'] = np.sin(self.train_raw['time']%(24*60*60*1000)*(2*np.pi/(24*60*60*1000)))
#         self.train_transformed['time'] = self.train_raw['time']

        #Now make dictionaries from the new variable names to the old and back
        
        # First sensor name to id list
        # First the numerical sensors
        sen_to_idx_list = [] # Will make a dictionary out of this
        for i in range(len(self.num_sensors)):
            transCols = list(self.train_transformed.columns)
            sen_to_idx_list = sen_to_idx_list + [(self.num_sensors[i],[transCols[i]])]
        # Now the categorical sensors
        base = len(self.num_sensors)
        for x in self.cat_sensors:
            newpair = (x, list(range(base,base+self.fittingdf[x].nunique())))
            sen_to_idx_list.append(newpair)
            base+= self.fittingdf[x].nunique()
        # The dictionary mapping sensor id to the list of indices in the transformed data that represent it
        self.sen_to_idx_dict = dict(sen_to_idx_list)
        
        #Now use the dictionary to get a list of the ids for the non-categorical and categorical sensors
        self.num_id_list = [x[0] for x in self.sen_to_idx_dict.values() if len(x) ==1]
        self.cat_id_list = [x[0] for x in self.sen_to_idx_dict.values() if len(x) > 1]
        
        
        # Next making the reverse dictionary from index to snesor id
        idx_to_sen_list = []
        for pair in sen_to_idx_list:
            sublist = [(idx, pair[0]) for idx in pair[1]]
            idx_to_sen_list = idx_to_sen_list + sublist
        self.idx_to_sen_dict = dict(idx_to_sen_list)
        
        # Now make a data frame to hold aggregate values for each numerical sensor (R^2, etc)
        self.agg_df = pd.DataFrame(index=self.num_sensors)
        
        # Finally make the Keras sequence generators
        # Make the Keras sequence generator for the training data
        train_n = int(len(self.train_transformed)*(1-train_val_ratio))
        val_n = len(self.train_transformed) - train_n
        self.train_gen = TimeseriesGenerator(self.train_transformed.iloc[:train_n,:].values,
                                        self.train_transformed[self.num_id_list].iloc[:train_n,:].values,
                                        length=self.windowL, sampling_rate=1,stride=1,
                                        batch_size=self.batch_size)
        # Make the Keras sequence generator for training time validation data
        self.train_val_gen = TimeseriesGenerator(self.train_transformed.iloc[train_n:,:].values,
                                        self.train_transformed[self.num_id_list].iloc[train_n:,:].values,
                                        length=self.windowL, sampling_rate=1,stride=1,
                                        batch_size=self.batch_size)

        
        
    def preprocess(self, data):
        data_index = data.index.tolist()
        # Difference the sensors highly correlated with time
        timedf = pd.DataFrame(index=data.index)
        timedf['time'] = data.time
        for sensor in self.timeSensors:
            if sensor == 'time':
                continue
            parts = re.split(r'\_\_',sensor) #JCSAT NAMING CONVENTIONS
#             parts = re.split(r':',sensor) #AM10 NAMING CONVENTIONS
            diffname = parts[0]+'_DIFF__'+parts[1]
            timedf[diffname]=data[sensor].diff()
        timedf.fillna(method='bfill', inplace=True)
        # Drop diff'ed sensors correlated with time (and drop time)
        timedf.drop(self.difftimeSensors, axis=1, inplace=True)

        frames = [data[list(set(self.ids)-set(self.timeSensors))], timedf]
        fittingdf = pd.concat(frames, axis=1)
        trans = self.preprocessor.transform(fittingdf) #transformed training data
        try:
            data_transformed = pd.DataFrame(trans.todense(),index=self.test_index)#,index=self.test_raw.index)
        except:
            data_transformed = pd.DataFrame(trans)#,index=self.test_raw.index
        data_transformed['time'] = list(data['time'])
        data_transformed['day_time_x'] = np.cos(data_transformed['time']%(24*60*60*1000)*(2*np.pi/(24*60*60*1000)))
        data_transformed['day_time_y'] = np.sin(data_transformed['time']%(24*60*60*1000)*(2*np.pi/(24*60*60*1000)))
        data_transformed.drop(['time'],axis=1, inplace=True)
        return data_transformed
    
    def process_test(self, test_raw):
        self.oldPred = True
        self.test_raw = test_raw
        self.test_transformed = self.preprocess(test_raw)
        
        # Make the Keras sequence generator for the test data
        self.test_gen = TimeseriesGenerator(self.test_transformed.values,
                                            self.test_transformed[self.num_id_list].values,
                                            length=self.windowL, sampling_rate=1,stride=1,
                                            batch_size=1)
        
    def process_validation(self):
        self.validate_transformed = self.preprocess(self.validate_raw)
        
        # Make the Keras sequence generator for the validate data
        self.val_gen = TimeseriesGenerator(self.validate_transformed.values,
                                           self.validate_transformed[self.num_id_list].values,
                                           length=self.windowL, sampling_rate=1,stride=1,
                                           batch_size=1)
        
        
    def make_batch(self, bats):
        self.bats = bats #batch size
        self.batch_data = np.zeros((self.bats,self.windowL,len(self.train_transformed.columns)))
        self.batch_y = np.zeros((self.bats,len(self.num_id_list)))
        startinds = np.random.randint(0,len(self.train_transformed)-(self.windowL+1),bats)
#         print(startinds[:10])
        for ind, start_loc in enumerate(startinds):
            self.batch_data[ind,:,:] = self.train_transformed.iloc[start_loc:start_loc+self.windowL].values
            self.batch_y[ind:] = self.train_transformed[self.num_id_list].iloc[start_loc+self.windowL,:]
       
            
    def make_model(self, lstm_layers = 1, lstm_units = 1000, metric='mean_squared_error'):
        if lstm_layers == 1:
            sensN = len(self.train_transformed.columns) -2 # number of sensors (eliminating the two time ones)
            outN = len(self.num_id_list) # number of output sensors; the non-categorical ones
            self.model = Sequential()
            self.model.add(LSTM(units=lstm_units, input_shape = (self.windowL, sensN+2), return_sequences=False))
            self.model.add(Dense(units = outN, activation='linear'))
            self.model.compile(optimizer='adam', loss='mean_squared_error',metrics=[metric])
            print(self.model.summary())
        elif lstm_layers == 2:
            sensN = len(self.train_transformed.columns) -2 # number of sensors (eliminating the two time ones)
            self.model = Sequential()
            self.model.add(LSTM(units=lstm_units, input_shape = (self.windowL, sensN+2), return_sequences=True))
            self.model.add(Dropout(0.4))
            self.model.add(LSTM(units=lstm_units, return_sequences=False))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(units = outN, activation='linear'))
            self.model.compile(optimizer='adam', loss='mean_squared_error',metrics=[metric])
            print(self.model.summary())
        else:
            print('lstm_layers must be 1 or 2')
        
    def fit_model(self, epochs = 3, verbose=True):
        self.oldPred = True
        self.model.fit_generator(self.train_gen,
                       validation_data = self.train_val_gen,
                       epochs = epochs,
                       verbose=verbose)
        self.train_loss = self.train_loss + self.model.history.history['loss']
        self.val_loss = self.val_loss + self.model.history.history['val_loss']

        
    def predict_train(self):
        """
        Predict on the training set for the purpose of determining which sensors are well-predictable.
        """
        self.train_pred = True
        self.train_preds = self.model.predict_generator(self.train_gen)
        # Because the validation set is taken from the training set before scaling and training, this must be cut.
        self.train_y = self.train_transformed[self.num_id_list].values[self.windowL:len(self.train_preds)+self.windowL]

        self.train_error_var = np.var(self.train_y - self.train_preds, axis=0) # variance of the predictions error on training
        self.train_error_mean = np.mean(self.train_y - self.train_preds, axis=0) # mean of the prediction error on training
        self.train_abserrordf = pd.DataFrame(np.abs(self.train_y - self.train_preds), columns=self.num_id_list)
        self.train_abs_errordf = pd.DataFrame(np.abs(self.train_y - self.train_preds), columns=self.num_id_list)
        self.train_errordf = pd.DataFrame((self.train_y - self.train_preds), columns=self.num_id_list)
        
        
    def predict_validate(self):
        """
        Predict on the validation set for the purpose of determining which sensors are well-predictable.
        """
        self.validate_pred = True
        self.validate_preds = self.model.predict_generator(self.val_gen)
        self.validate_y = self.validate_transformed[self.num_id_list].values[self.windowL:]
        
        self.r2 = r2_score(self.validate_y, self.validate_preds, multioutput='raw_values')
        self.agg_df['validate_r2'] = self.r2
        self.validate_error_var = np.var(self.validate_y - self.validate_preds, axis=0) # variance of the predictions error on training
        self.agg_df['validate_error_var'] = self.validate_error_var
        self.validate_error_mean = np.mean(self.validate_y - self.validate_preds, axis=0) # mean of the prediction error on training
        self.agg_df['validate_error_mean'] = self.validate_error_mean
#        self.validate_abserrordf = pd.DataFrame(np.abs(self.validate_y - self.validate_preds), columns=self.num_id_list)
        self.validate_abs_errordf = pd.DataFrame(np.abs(self.validate_y - self.validate_preds), columns=self.num_id_list)
        self.validate_errordf = pd.DataFrame((self.validate_y - self.validate_preds), columns=self.num_id_list)
     
     
         
       
    def predict_new(self):
        self.oldPred = False
        self.preds = self.model.predict_generator(self.test_gen)
        self.test_y = self.test_transformed[self.num_id_list].values[self.windowL:]
        self.test_abs_errordf = pd.DataFrame(np.abs(self.test_y - self.preds), columns=self.num_id_list)
        self.test_errordf = pd.DataFrame((self.test_y - self.preds), columns=self.num_id_list)
        
          

            
    def score_test(self, r2_threshold = -np.inf):
        self.r2_threshold = r2_threshold # # R^2 cutoff for using a sensor in score, etc.
#         self.high_quality_sensors2 = [x for (i,x) in enumerate(self.num_id_list) if self.r2[i] >= self.r2_threshold]
        self.high_quality_sensor_names = [x for x in self.agg_df.index.tolist() if self.agg_df.loc[x,'validate_r2'] >= self.r2_threshold]
        self.high_quality_sensors = [self.sen_to_idx_dict[x][0] for x in self.high_quality_sensor_names]
        self.high_qualitypred_df = pd.DataFrame(self.preds, columns=self.num_id_list)[self.high_quality_sensors]
        if self.oldPred:
            print('Predictions are old, so re-predicting.')
            self.predict_new()
        high_quality_y = pd.DataFrame(self.test_y, columns = self.num_id_list)[self.high_quality_sensors]
        self.score = np.linalg.norm(self.high_qualitypred_df.values - high_quality_y.values,axis=1) #score not scaled by R^2
        
        # For R^2 scaling for the score
        
        error_mat = np.abs(self.high_qualitypred_df.values - high_quality_y.values)
        scaled_error_mat = error_mat * self.r2[np.where(self.r2 > self.r2_threshold)].reshape(1,-1)
        self.r2_score = np.sum(scaled_error_mat, axis=1)
        
    def score_validate(self):
        high_qualitypred_df = pd.DataFrame(self.validate_preds, columns=self.num_id_list)[self.high_quality_sensors]
        high_quality_y = pd.DataFrame(self.validate_y, columns = self.num_id_list)[self.high_quality_sensors]
        self.valid_score = np.linalg.norm(high_qualitypred_df.values - high_quality_y.values,axis=1) #score not scaled by R^2
        
        # For R^2 scaling for the score
        
        error_mat = np.abs(high_qualitypred_df.values - high_quality_y.values)
        scaled_error_mat = error_mat * self.r2[np.where(self.r2 > self.r2_threshold)].reshape(1,-1)
        self.r2_valid_score = np.sum(scaled_error_mat, axis=1)
          
       
