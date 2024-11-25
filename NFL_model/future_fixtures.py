#!/usr/bin/env python
# coding: utf-8

# In[270]:


import pandas as pd
import scipy.stats as stats
from datetime import datetime, timedelta
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import statsmodels.api as sm
import importlib
import NFL2
importlib.reload(NFL2)
from itertools import product
import warnings


from NFL2 import (correlationdistance, home_cover_percentage, away_cover_percentage, gap_final_results, overtime_final_results,spreads, dome_teams_away,
dome_teams, cold_teams, hot_teams, predict_next_score, team_distances,spreads)
df_distance = pd.read_pickle("df_distance.pkl")


# In[261]:


# File Path

file_path = r'D:\NFL.xlsx'
df = pd.read_excel(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Push'] == '-']
start_date = datetime.strptime('2014-09-05', '%Y-%m-%d')
df_new = df[df['Date'] > (start_date +timedelta(days=6)) ]
df_old = df[df['Date'] < (start_date +timedelta(days=6)) ]
#print(df_new.shape[0])


# In[262]:


# DISTANCES

df_future = df_new[['Home Team', 'Away Team']]
pd.set_option('display.float_format', '{:.1f}'.format)

def final():

    # Step 1: Remove duplicate rows based on 'Home Team' and 'Away Team'
    distances_unique = team_distances.drop_duplicates(subset=['Home Team', 'Away Team'])

    # From regression model
    m = 0.0028 
    c = 45.2925
    distances_unique['Home Cover (dist) %'] = distances_unique['Distance'] * m + c
    
    df_merged = pd.merge(df_future, distances_unique, on=['Home Team', 'Away Team'], how='left')
    df_merged['Away Cover (dist) %'] = 100 - df_merged['Home Cover (dist) %']
    
    df_merged['Home Cover'] = home_cover_percentage
    df_merged['Away Cover'] = away_cover_percentage

    df_final = df_merged[['Home Team','Away Team','Home Cover','Away Cover','Distance','Home Cover (dist) %','Away Cover (dist) %']]
    df_merged['+'] = df_merged['Home Cover'] + df_merged['Away Cover']
    return df_final

df_final = final()

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)



# In[263]:


# GAP For Future Games

df = pd.read_excel(file_path)
df = df[df['Push'] == 1]
df['Date'] = pd.to_datetime(df['Date'])

df = df[df['Date'] > start_date].sort_values(by='Date')

df_gap = gap_final_results[gap_final_results['game_count'] > 40]
df_gap['Team Type Win/Loss'] = df_gap['Team Type'] + '-' + df_gap['Win/Loss']



def gap_get_most_recent_game_info(team, df, current_date):
    home_games = df[(df['Home Team'] == team) & (df['Date'] < current_date)]
    away_games = df[(df['Away Team'] == team) & (df['Date'] < current_date)]
    
    all_games = pd.concat([home_games, away_games])
    if all_games.empty:
        return None
    
    # Find the most recent game
    most_recent_game = all_games.sort_values(by='Date', ascending=False).iloc[0]
    
    game_type = 'Home' if most_recent_game['Home Team'] == team else 'Away'
    
    if game_type == 'Home':
        win_loss = 'Win' if most_recent_game['Spread Covered Home'] == 1 else 'Loss'
    else:
        win_loss = 'Win' if most_recent_game['Spread Covered Away'] == 1 else 'Loss'
    
    return {
        'Date': most_recent_game['Date'],
        'Game Type': game_type,
        'Win/Loss': win_loss
    }



results = []

for index, row in df.iterrows():
    current_date = row['Date']
    home_team = row['Home Team']
    away_team = row['Away Team']

    # Recent game
    home_recent = gap_get_most_recent_game_info(home_team, df, current_date)
    away_recent = gap_get_most_recent_game_info(away_team, df, current_date)
    
    if home_recent and away_recent:
        home_gap = (current_date - home_recent['Date']).days
        away_gap = (current_date - away_recent['Date']).days
        
        future_home_game_type = 'Home'
        future_away_game_type = 'Away'
    
        home_game_type_full = f"{home_recent['Game Type']}-{future_home_game_type}"
        away_game_type_full = f"{away_recent['Game Type']}-{future_away_game_type}"
        
        home_match = df_gap[
            (df_gap['Gap (Days)'] == home_gap) &
            (df_gap['Team Type'] == home_game_type_full) &
            (df_gap['Win/Loss'] == home_recent['Win/Loss'])
        ]
        
        away_match = df_gap[
            (df_gap['Gap (Days)'] == away_gap) &
            (df_gap['Team Type'] == away_game_type_full) &
            (df_gap['Win/Loss'] == away_recent['Win/Loss'])
        ]
        
        home_percentage = home_match['Percentage Covered'].iloc[0] if not home_match.empty else None
        away_percentage = away_match['Percentage Covered'].iloc[0] if not away_match.empty else None
        
        results.append({
            'Future Game Date': current_date,
            'Home Team': home_team,
            'Away Team': away_team,
            'Home Team Most Recent Game Date': home_recent['Date'],
            'Home Game Type': home_game_type_full,
            'Home Win/Loss': home_recent['Win/Loss'],
            'Home Gap': home_gap,
            'Home Percentage Covered': home_percentage,
            'Away Team Most Recent Game Date': away_recent['Date'],
            'Away Game Type': away_game_type_full,
            'Away Win/Loss': away_recent['Win/Loss'],
            'Away Gap': away_gap,
            'Away Percentage Covered': away_percentage
        })

# Results
df_results = pd.DataFrame(results)


# In[264]:


# GAP 2
# This section find the previous and next game types (away-home, home-away) and pairs it with win/loss and gap between games

# Selects instances with game count > 40

pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  

def transform_gap_key(row):
    if "Home" in row['Team Type']:
        home_away = "Home"
    else:
        home_away = "Away"
    
    if row['Team Type'] == "Home-H to A":
        home_away_type = "Away"
    elif row['Team Type'] == "Home-H to H":
        home_away_type = "Home"
    elif row['Team Type'] == "Away-A to H":
        home_away_type = "Home"
    elif row['Team Type'] == "Away-A to A":
        home_away_type = "Away"
    
    return f"{home_away}-{home_away_type}"

df_gap['key_transformer'] = df_gap.apply(transform_gap_key, axis=1)
df_gap['key_transformed'] = df_gap['Gap (Days)'].astype(str) + '-' + df_gap['key_transformer'] + '-' + df_gap['Win/Loss']

df_gap_unique = df_gap.groupby('key_transformed', as_index=False)['Percentage Covered'].mean()

df_new['home_key'] = df_results['Home Gap'].astype(str) + '-' + df_results['Home Game Type'] + '-' + df_results['Home Win/Loss']
df_new['away_key'] = df_results['Away Gap'].astype(str) + '-' + df_results['Away Game Type'] + '-' + df_results['Away Win/Loss']

# Results
df_new['Home Percentage Covered'] = df_new['home_key'].map(df_gap_unique.set_index('key_transformed')['Percentage Covered'])
df_new['Away Percentage Covered'] = df_new['away_key'].map(df_gap_unique.set_index('key_transformed')['Percentage Covered'])

df_new['Home Percentage Covered'] = df_new['Home Percentage Covered'].fillna(df_final['Home Cover'])
df_new['Away Percentage Covered'] = df_new['Away Percentage Covered'].fillna(df_final['Away Cover'])


df_final['H Gap'] = df_new['Home Percentage Covered']
df_final['A Gap'] = df_new['Away Percentage Covered']


# In[265]:


# OVERTIME

overtime_final_results_gc = overtime_final_results[overtime_final_results['game_count'] >= 15]

# Previous Results from NFL2
ot_results = {
    'Gap (Days)': [7, 7, 7, 7],
    'Team Type': ['Away-Away', 'Away-Home', 'Home-Away', 'Home-Home'],
    'game_count': [43, 57, 68, 30],
    'covered_count': [24, 26, 31, 7],
    'Percentage Covered': [55.8, 45.6, 45.6, 23.3]
}
ot_results_df = pd.DataFrame(ot_results)


df = pd.read_excel(file_path)
df = df[df['Push'] == 1]
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] > start_date]
df = df.sort_values(by='Date')

# Find out if a team went to Overtime in their previous game
def get_most_recent_game_info(team, df, current_date):
    home_games = df[(df['Home Team'] == team) & (df['Date'] < current_date)]
    away_games = df[(df['Away Team'] == team) & (df['Date'] < current_date)]
    
    all_games = pd.concat([home_games, away_games])
    if all_games.empty:
        return None, None, None
    
    most_recent_game = all_games.sort_values(by='Date', ascending=False).iloc[0]
    if most_recent_game['Home Team'] == team:
        game_type = 'Home'
    else:
        game_type = 'Away'
    
    return most_recent_game['Date'], game_type, most_recent_game['Overtime?']

def get_recent_games_for_teams(df):
    results = []
    
    for index, row in df.iterrows():
        future_game_date = row['Date']
        home_team = row['Home Team']
        away_team = row['Away Team']
        
        home_team_most_recent_date, home_game_type, home_team_overtime = get_most_recent_game_info(home_team, df, future_game_date)
        away_team_most_recent_date, away_game_type, away_team_overtime = get_most_recent_game_info(away_team, df, future_game_date)
        
        if home_team_most_recent_date is None or away_team_most_recent_date is None:
            continue  
        future_home_game_type = 'Home' if row['Home Team'] == home_team else 'Away'
        future_away_game_type = 'Away' if row['Away Team'] == away_team else 'Home'
        
        home_game_type_full = f"{home_game_type}-{future_home_game_type}"
        away_game_type_full = f"{away_game_type}-{future_away_game_type}"
        
        results.append({
            'Future Game Date': future_game_date,
            'Home Team': home_team,
            'Away Team': away_team,
            'Home Team Most Recent Game Date': home_team_most_recent_date,
            'Home Game Type': home_game_type_full,  
            'Home Team Overtime': home_team_overtime,
            'Away Team Most Recent Game Date': away_team_most_recent_date,
            'Away Game Type': away_game_type_full, 
            'Away Team Overtime': away_team_overtime
        })
    
    df_result = pd.DataFrame(results)
    return df_result

df_with_recent_games = get_recent_games_for_teams(df)

# Results
df_with_recent_games['Home OT'] = df_with_recent_games.apply(
    lambda row: ot_results_df.loc[ot_results_df['Team Type'] == row['Home Game Type'], 'Percentage Covered'].values[0]
    if row['Home Team Overtime'] == 'Y' else None,
    axis=1
)

df_with_recent_games['Away OT'] = df_with_recent_games.apply(
    lambda row: ot_results_df.loc[ot_results_df['Team Type'] == row['Away Game Type'], 'Percentage Covered'].values[0]
    if row['Away Team Overtime'] == 'Y' else None,
    axis=1
)


# In[266]:


### SPREADS

from NFL2 import spread_success
from NFL2 import spreads

spreads = spread_success[(spread_success[('Bet Success Home', 'count')] >= 30)]

spreads['H Spread Cover %'] = spreads['Cover %']
spreads['A Spread Cover %'] = 100 - spreads['H Spread Cover %']

# Maps % cover for each spread to the data in the file
df_new['H Spread Cover Match'] = df_new['Home Line Open'].map(spreads['H Spread Cover %'])
df_new['A Spread Cover Match'] = df_new['Away Line Open'].map(spreads['A Spread Cover %'])

# Fills N/A with Home/Away Cover % found earlier
df_new['H Spread Cover'] = df_new['H Spread Cover Match'].fillna(df_final['Home Cover'])
df_new['A Spread Cover'] = 100 - df_new['H Spread Cover Match'].fillna(df_final['Away Cover'])

# Experiment with weights to reduce the impact of this metric
weight = 0.05
df_new['H Spread Cover'] = weight * 50 + (1 - weight) * df_new['H Spread Cover']
df_new['A Spread Cover'] = weight * 50 + (1 - weight) * df_new['A Spread Cover']


df_final['H Spread Cover'] = df_new['H Spread Cover']
df_final['A Spread Cover'] = df_new['A Spread Cover']


# In[267]:


### DOME, HOT, COLD

filtered_df, cold_teams_cover_dome, dome_teams_cover_cold, hot_teams_cover_cold, cold_teams_cover_hot = dome_teams_away()

df_domecold = pd.DataFrame()
df_domecold['Home vs Dome/Hot'] = np.where(df_new['Away Team'].isin(dome_teams) & (df_new['Home Team'].isin(cold_teams)), cold_teams_cover_dome*100, df_final['Home Cover'])
df_domecold['Away vs Cold'] = np.where(df_new['Away Team'].isin(dome_teams) & (df_new['Home Team'].isin(cold_teams)), dome_teams_cover_cold*100, df_final['Away Cover'])

df_domecold['Home vs Dome/Hot'] = np.where(df_new['Away Team'].isin(hot_teams) & (df_new['Home Team'].isin(cold_teams)), cold_teams_cover_hot*100, df_final['Home Cover'])
df_domecold['Away vs Cold'] = np.where(df_new['Away Team'].isin(hot_teams) & (df_new['Home Team'].isin(cold_teams)), hot_teams_cover_cold*100, df_final['Away Cover'])

df_domecold['Home Team'] = df_new['Home Team']
df_domecold['Away Team'] = df_new['Away Team']


# In[268]:


### GROUPING OF METRICS

### METRICS ADDED: home/away, OT, Away distance, spreads, dome/hot/cold, Gap between games + win/loss + H/A

df_final['Home OT'] = df_with_recent_games['Home OT']
df_final['Away OT'] = df_with_recent_games['Away OT']

df_final['Home OT + Away'] = df_final.apply(lambda row: row['Home OT'] if pd.notna(row['Home OT']) else (100 - row['Away OT'] if pd.notna(row['Away OT']) else None), axis=1)
df_final['Away OT + Home'] = df_final.apply(lambda row: row['Away OT'] if pd.notna(row['Away OT']) else (100 - row['Home OT'] if pd.notna(row['Home OT']) else None), axis=1)

df_final['Home OT + Away'] = df_final['Home OT + Away'].fillna(df_final['Home Cover'])
df_final['Away OT + Home'] = df_final['Away OT + Home'].fillna(df_final['Away Cover'])

df_final['Home vs Dome/Hot'] = df_domecold['Home vs Dome/Hot']
df_final['Away vs Cold'] = df_domecold['Away vs Cold']

df_final['A Spread Cover'] = df_final['A Spread Cover'] * (100/(df_final['A Spread Cover']+df_final['H Spread Cover']))
df_final['H Spread Cover'] = df_final['H Spread Cover'] * (100/(df_final['A Spread Cover']+df_final['H Spread Cover']))

#Best weight combination: (0.1, 0.1, 0.1, 0.4, 0.1, 0.2) - 20% select rate - 61% win rate

df_final['Home Cover %']  = (df_final['Home Cover'] * 0.1) + (df_final['Home Cover (dist) %'] * 0.1) + (df_final['Home OT + Away'] * 0.1
                                                                                                       + df_final['H Spread Cover'] * 0.4
                                                                                                       + df_final['Home vs Dome/Hot'] * 0.1
                                                                                                         + df_final['H Gap'] * 0.2)
df_final['Away Cover %']  = (df_final['Away Cover'] * 0.1) + (df_final['Away Cover (dist) %'] * 0.1) + (df_final['Away OT + Home'] * 0.1
                                                                                                       + df_final['A Spread Cover'] * 0.4
                                                                                                       + df_final['Away vs Cold'] * 0.1
                                                                                                       + df_final['A Gap'] * 0.2)

# Select games where one team has 53% chance of covering the spread
df_final['Cover?'] = df_final['Home Cover %'].apply(lambda x: 'Yes' if x > 53 else ('Yes' if x < 47 else 'No'))



df_sorted_asc = df_final.sort_values(by='Home Cover %', ascending=True)

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


# In[269]:


# FIND CORRECT WEIGHTS FOR MODEL

# Weight range for the optimization
weight_range = np.arange(0.1, 0.5, 0.1)


# Create all possible weight combinations (since we have 5 pairs of equal weights, we only need to iterate over 5 parameters)
weight_combinations = list(product(weight_range, repeat=6))

valid_combinations = [weights for weights in weight_combinations if sum(weights) == 1]

# Initialize variables to track the best combination
best_combination = None
best_win_count = 0

# Loop through all weight combinations
best = []
k = 10

for combination in valid_combinations:
    # Unpack the combination to apply to the different pairs of columns
    h_cover_weight, h_dist_weight, h_ot_weight, h_spread_weight, h_dome_weight, h_gap = combination
    
    # Calculate the Home and Away Cover % using the same weights for the corresponding columns
    df_final['Home Cover %'] = (df_final['Home Cover'] * h_cover_weight) + \
                                    (df_final['Home Cover (dist) %'] * h_dist_weight) + \
                                    (df_final['Home OT + Away'] * h_ot_weight) + \
                                    (df_final['H Spread Cover'] * h_spread_weight) + \
                                    (df_final['Home vs Dome/Hot'] * h_dome_weight) + \
                                    (df_final['H Gap'] * h_gap)
        
    df_final['Away Cover %'] = (df_final['Away Cover'] * h_cover_weight) + \
                                    (df_final['Away Cover (dist) %'] * h_dist_weight) + \
                                    (df_final['Away OT + Home'] * h_ot_weight) + \
                                    (df_final['A Spread Cover'] * h_spread_weight) + \
                                    (df_final['Away vs Cold'] * h_dome_weight) + \
                                    (df_final['A Gap'] * h_gap)
    
    
    # Recalculate the Cover success
    df_final['Cover?'] = df_final['Home Cover %'].apply(lambda x: 'Yes' if x > 53 else ('Yes' if x < 47 else 'No'))
    df_cover = df_final[df_final['Cover?'] == "Yes"]
    
    #df_cover['Home Spread'] = df_new['Home Line Open']
    df_cover.loc[:, 'Home Spread'] = df_new['Home Line Open']
    df_cover.loc[:, 'SCH'] = df_new['Spread Covered Home']
    df_cover.loc[:, 'SCA'] = df_new['Spread Covered Home']
    #df_cover['SCH'] = df_new['Spread Covered Away']
    #df_cover['SCA'] = df_new['Spread Covered Away']

    df_cover.loc[:, 'Cover?'] = df_cover['Home Cover %'].apply(lambda x: 'Home' if x > 53 else 'Away')
    df_cover.loc[:,'Successful?'] = df_cover.apply(lambda row: 0 if row['Cover?'] == 'Away' and row['SCH'] == 1 else 1, axis=1)
    
    # Calculate win count
    win_count = (df_cover['Successful?'] == 1).sum()

    '''if (win_count/len(df_cover) > 0.6) & (len(df_cover) > 70):
        
        if win_count > best_win_count:
            best = [combination, len(df_cover), win_count,win_count/len(df_cover)]
            best_win_count = win_count'''

            

'''# Output the best weight combination and its corresponding win count
print(f"Best weight combination: {best[0]}")
print(f"Highest win count: {best[2]}")
print(f"Amount: {best[1]}")
print(f"{round(100*best[2]/best[1],ndigits=0)}%")'''


# In[ ]:





# In[ ]:




