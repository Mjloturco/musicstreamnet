"""
Read in SpotGenTrack dataset and Philippenes Daily Top 200 Dataset 

"""

import numpy as np 
import pandas as pd 

# define paths to data 
SPOTGEN_PATH = '../data/spotify_tracks.csv'
PHIL_DAILY_PATH = '../data/spotify_daily_charts.csv'

DST_PATH = '../data/joined.csv'


def joinSpotifyDatasets(data1, data2):
    """
    join datasets and clean up column names
    """
    
    joined = data1.join(data2, how='inner')
    
    # drop redundant/unnecessary columns
    to_drop = ['Unnamed: 0', 'country', 'disc_number', 'track_name_prev', 'track_href', 'name', 'playlist', 'mode']
    
    joined = joined.drop(columns=to_drop)
    
    return joined 
    
    

if __name__ == "__main__":
     
    # Read in the datasets 
    spotgen = pd.read_csv(SPOTGEN_PATH).set_index('id')
    phil_daily = pd.read_csv(PHIL_DAILY_PATH).set_index('track_id')
    
    # join dataframes 
    joined = joinSpotifyDatasets(spotgen, phil_daily)

    # write joined dataframe as .csv
    joined.to_csv(DST_PATH, sep=',')