import os 
import numpy as np 
import pandas as pd 
from tensorflow.keras.layers import Input, LSTM, concatenate
from tensorflow.
# import keras 

# SPOTGEN_PATH = '../data/spotify_tracks.csv'
PHIL_DAILY_PATH = 'data/spotify_daily_charts.csv'

# set destination path 
TRACK_PATH = 'data/spotify_tracks_augmented.csv'
AUG_PATH = 'data/spotify_daily_augmented.csv'
SONG_MAJOR_PATH = 'data/spotify_daily_song_major.csv'


TIME_SERIES_OUTPUT_SIZE = 16
if __name__ == "__main__":   


    #CLEAN IT UP
    song_major = pd.read_csv(SONG_MAJOR_PATH)
    song_major.sort_values(by="track_id", ascending=False, inplace=True)
    song_major.drop(columns=["track_id", "stream_totals", "track_name"], inplace=True)
    split_point = int(3495*.8)
    time_series_train = song_major.iloc[:split_point,:]
    time_series_test = song_major.iloc[split_point+1:,:]


    with_meta = pd.read_csv(TRACK_PATH)
    with_meta.sort_values(by="track_id", ascending=False, inplace=True)
    with_meta.drop(columns=["track_id", "artist_id", "uri", "type", "track_href", "analysis_url", "time_signature", "id"], inplace=True)
    meta_train = with_meta.iloc[:split_point, :]
    meta_test = with_meta.iloc[split_point:, :]

    # print(meta_train.iloc[0,:].to_string())
    # print(time_series_train.iloc[0,:].to_string())
    # Create META input
    meta_input = Input(shape=(12,), name='Metadata Input Layer')


    time_series_input = Input(shape=(1600,), name="Time Input Layer")
    lstm_out = LSTM(TIME_SERIES_OUTPUT_SIZE)(time_series_input)
    full_features = concatenate([lstm_out, meta_input])

    d1 = Dense(12, activation='relu')(full_features)
    d2 = Dense(8, )

    #output
    output = Dense(1, )