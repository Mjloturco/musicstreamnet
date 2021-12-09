"""
read in Philippenes Daily Top 200 Dataset 
use Spotify API to get additional features that are in SpotGen data 
save new data file

"""
import os 
import numpy as np 
import pandas as pd 

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

"""
Spotify API authentication
NB: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET need to be defined as 
environment variables
"""
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                            client_id=os.getenv('SPOTIFY_CLIENT_ID'),
                            client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')))

"""
Define input and output data paths  
"""
# SPOTGEN_PATH = '../data/spotify_tracks.csv'
PHIL_DAILY_PATH = 'data/spotify_daily_charts.csv'

# set destination path 
TRACK_PATH = '../data/spotify_tracks_augmented.csv'
AUG_PATH = '../data/spotify_daily_augmented.csv'
SONG_MAJOR_PATH = 'data/spotify_daily_song_major.csv'


def getAdditionalFeatures(df):
    """
    Get features for each track from the Spotify API and join them to 
    the Philippenes Daily dataframe
    """
    
    # artist ID 
    df['artist_id'] = df.apply(
                        lambda r:sp.track(r['track_id']) ['artists'][0]['id'],
                        axis=1)
        
    # list of features to extract from audio_features dictionary
    features = ['acousticness', 'danceability', 'duration_ms',
                'instrumentalness', 'liveness', 'loudness', 'speechiness',
                'tempo', 'valence', 'key', 'time_signature']
                
    # extract audio features 
    df['audiofeats'] = df.apply(
                        lambda row: sp.audio_features(row['track_id'])[0],  
                        axis=1)
    
    # explode feature dictionary into separate columns
    df = pd.concat([df.drop(['audiofeats'], axis=1),
                    df['audiofeats'].apply(pd.Series)], axis=1)

    
    return df 

def produceSongMajor():
    phil_daily = pd.read_csv(PHIL_DAILY_PATH, parse_dates=["date"])

    date_order = phil_daily.pivot(index=['track_id', 'track_name'], columns="date", values="streams")

    date_order = date_order.fillna(0)

    date_order["stream_totals"] = date_order.iloc[:, :].sum(axis=1)
    date_order.sort_values(by="stream_totals", ascending=False, inplace=True)

    # print(date_order.head(50))
    # print(date_order.iloc[[2600]].to_string())
    date_order.to_csv(SONG_MAJOR_PATH, sep=',')

   


def joinSpotifyDatasets(data1, data2):
    """
    DON'T USE THIS! 
    join datasets and clean up column names
    """
    
    joined = data1.join(data2, how='inner')
    
    # drop redundant/unnecessary columns
    to_drop = ['Unnamed: 0', 'country', 'disc_number', 'track_name_prev', 'track_href', 'name', 'playlist', 'mode']
    
    joined = joined.drop(columns=to_drop)
    
    return joined 

if __name__ == "__main__":
     
    # produceSongMajor()
    # Read in the datasets 
    phil_daily = pd.read_csv(PHIL_DAILY_PATH)
    
    # get unique track ids and drop redundant columns
    tracks = phil_daily.groupby(by='track_id', as_index=False).max()
    tracks = tracks.drop(['date', 'position', 'track_name', 
                            'artist', 'streams'], axis=1)
    with_features = getAdditionalFeatures(tracks)
        
    # save this intermediate dataset
    with_features.to_csv(TRACK_PATH, sep=',')
    
    # join audio features data with original daily tracks data 
    augmented = with_features.set_index('track_id') \
                             .join(phil_daily.set_index('track_id'))
    

    # write joined dataframe as .csv
    augmented.to_csv(AUG_PATH, sep=',')