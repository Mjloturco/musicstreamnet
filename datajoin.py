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
PHIL_DAILY_PATH = '../data/spotify_daily_charts.csv'

# set destination path 
DST_PATH = '../data/spotify_daily_augmented.csv'


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
     
    # Read in the datasets 
    phil_daily = pd.read_csv(PHIL_DAILY_PATH)
    phil_daily = phil_daily.head(1000) # FIX: API times out
    
    augmented = getAdditionalFeatures(phil_daily)

    # write joined dataframe as .csv
    augmented.to_csv(DST_PATH, sep=',')