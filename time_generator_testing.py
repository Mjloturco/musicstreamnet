import os 
import numpy as np 
import pandas as pd 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate
# from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf 
# import keras 



# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)

# SPOTGEN_PATH = '../data/spotify_tracks.csv'
PHIL_DAILY_PATH = 'data/spotify_daily_charts.csv'

# set destination path 
TRACK_PATH = 'data/spotify_tracks_augmented.csv'
AUG_PATH = 'data/spotify_daily_augmented.csv'
SONG_MAJOR_PATH = 'data/spotify_daily_song_major.csv'


TIME_SERIES_OUTPUT_SIZE = 16
LOOK_BACK = 4
NSONGS = 3495

if __name__ == "__main__":   

    """
    Time series pre-processing
    """
    
    song_major = pd.read_csv(SONG_MAJOR_PATH)
    song_major.sort_values(by="track_id", ascending=False, inplace=True)
    song_major.drop(columns=["track_id", "stream_totals", "track_name"],        
                    inplace=True)
                    
    split_point = int(3495*.8)
    time_series_train = song_major.iloc[:split_point,:].to_numpy()
    time_series_test = song_major.iloc[split_point:,:].to_numpy()
    
    
    train_generator = TimeseriesGenerator(time_series_train, time_series_train,
                                            length=LOOK_BACK, batch_size=20) 
    test_generator = TimeseriesGenerator(time_series_test, time_series_test, 
                                        length=LOOK_BACK, batch_size=20)
    
    # # swap axes 
    # time_series_train = np.swapaxes(time_series_train, 0, 1)
    # time_series_test = np.swapaxes(time_series_test, 0, 1)
    # 
    # time_series_train = timeseries_dataset_from_array(time_series_train,  
    #                         None, sequence_length=LOOK_BACK,
    #                         batch_size=1600).as_numpy_iterator()
    # time_series_test = timeseries_dataset_from_array(time_series_test, 
    #                         None, sequence_length=LOOK_BACK,
    #                         batch_size=1600).as_numpy_iterator()
    
    
    
    # print(time_series_train.element_spec)
    # for batch in time_series_train:
    #     print('BATCH: ', batch)
    # time_series_train = np.swapaxes(time_series_train, 1, 2)
    # time_series_test = np.swapaxes(time_series_test[0], 1, 2)
    
    
    """
    Construct & train model 
    """
    # time_series_input = Input(shape=(1598, 1), name="Time Input Layer")
    time_series_input = Input(shape=(1598, 1))
    
    lstm_out = LSTM(TIME_SERIES_OUTPUT_SIZE)(time_series_input)
    
    # concatenate
    # full_features = concatenate([lstm_out, meta_input])

    # d1 = Dense(12, activation='relu')(full_features)
    d1 = Dense(12, activation='relu')(lstm_out)
    d2 = Dense(8, activation='relu')(d1)

    #outputshape
    output = Dense(1, activation='linear')(d2)
    
    # create model 
    model = Model(inputs=time_series_input, outputs=output)
    print(model.summary())
    
    # compile 
    model.compile(loss='mse', optimizer='adam')
    model.fit(x=train_generator, y=None, epochs=15, batch_size=None)
    
    
    
    