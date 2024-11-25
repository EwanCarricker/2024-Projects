#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import scipy.stats as stats
from datetime import datetime, timedelta
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import statsmodels.api as sm


# In[2]:


# File Path
file_path = r'D:\NFL.xlsx'
df = pd.read_excel(file_path)
df = df[df['Push'] == 1]


# In[3]:


### HOME AWAY COVERS

# Calculate if the Home team covered the spread (and it wasn't a push)
df['Home_Covered'] = (df['Home Score'] + df['Home Line Open']) > df['Away Score']

# Calculate if the Away team covered the spread (and it wasn't a push)
df['Away_Covered'] = df['Away Score'] > (df['Home Score'] + df['Home Line Open'])

# Calculate cover percentages
home_cover_percentage = df['Home_Covered'].sum() / df['Home_Covered'].shape[0] * 100
away_cover_percentage = df['Away_Covered'].sum() / df['Home_Covered'].shape[0] * 100

# Print the results
#print(f"Home Team Cover ATS: {home_cover_percentage:.2f}%")
#print(f"Away Team Cover ATS: {away_cover_percentage:.2f}%")


# In[4]:


### GAPS BETWEEN GAMES AFTER WINS/LOSSES COVERS

# List to store results
results = []
df = pd.read_excel(file_path)
df = df.sort_values(by='Date')
# Loop through each game
for index, row in df.iterrows():
    game_date = row['Date']
    home_team = row['Home Team']
    away_team = row['Away Team']
    home_score = row['Home Score']
    away_score = row['Away Score']

    # Determine the game outcome (Win or Loss)
    if home_score > away_score:
        home_team_result = 'Win'
        away_team_result = 'Loss'
    elif away_score > home_score:
        home_team_result = 'Loss'
        away_team_result = 'Win'
    else:
        home_team_result = 'Draw'
        away_team_result = 'Draw'

    # Find the next game for the home team (either as home or away)
    next_home_game_home = (
        df[(df['Home Team'] == home_team) & (df['Date'] > game_date)]
        .sort_values(by='Date')
        .iloc[0] if not df[(df['Home Team'] == home_team) & (df['Date'] > game_date)].empty else None
    )
    next_home_game_away = (
        df[(df['Away Team'] == home_team) & (df['Date'] > game_date)]
        .sort_values(by='Date')
        .iloc[0] if not df[(df['Away Team'] == home_team) & (df['Date'] > game_date)].empty else None
    )

    # Determine the earliest next game for the home team
    if next_home_game_home is not None and next_home_game_away is not None:
        next_home_game = next_home_game_home if next_home_game_home['Date'] < next_home_game_away['Date'] else next_home_game_away
    elif next_home_game_home is not None:
        next_home_game = next_home_game_home
    elif next_home_game_away is not None:
        next_home_game = next_home_game_away
    else:
        next_home_game = None

    # Find the next game for the away team (either as home or away)
    next_away_game_home = (
        df[(df['Home Team'] == away_team) & (df['Date'] > game_date)]
        .sort_values(by='Date')
        .iloc[0] if not df[(df['Home Team'] == away_team) & (df['Date'] > game_date)].empty else None
    )
    next_away_game_away = (
        df[(df['Away Team'] == away_team) & (df['Date'] > game_date)]
        .sort_values(by='Date')
        .iloc[0] if not df[(df['Away Team'] == away_team) & (df['Date'] > game_date)].empty else None
    )

    # Determine the earliest next game for the away team
    if next_away_game_home is not None and next_away_game_away is not None:
        next_away_game = next_away_game_home if next_away_game_home['Date'] < next_away_game_away['Date'] else next_away_game_away
    elif next_away_game_home is not None:
        next_away_game = next_away_game_home
    elif next_away_game_away is not None:
        next_away_game = next_away_game_away
    else:
        next_away_game = None

    # Calculate the gap in days and add to the results
    if next_home_game is not None:
        gap_days_home = (next_home_game['Date'] - game_date).days
        if next_home_game['Home Team'] == home_team:
            # Home team playing at home next
            results.append({
                'Gap (Days)': gap_days_home,
                'Team Type': 'Home-H to H',
                'game_count': 1,
                'covered_count': int(next_home_game['Spread Covered Home']),
                'Win/Loss': home_team_result
            })
        else:
            # Home team playing away next
            results.append({
                'Gap (Days)': gap_days_home,
                'Team Type': 'Home-H to A',
                'game_count': 1,
                'covered_count': int(next_home_game['Spread Covered Away']),
                'Win/Loss': home_team_result
            })

    if next_away_game is not None:
        gap_days_away = (next_away_game['Date'] - game_date).days
        if next_away_game['Home Team'] == away_team:
            # Away team playing at home next
            results.append({
                'Gap (Days)': gap_days_away,
                'Team Type': 'Away-A to H',
                'game_count': 1,
                'covered_count': int(next_away_game['Spread Covered Home']),
                'Win/Loss': away_team_result
            })
        else:
            # Away team playing away next
            results.append({
                'Gap (Days)': gap_days_away,
                'Team Type': 'Away-A to A',
                'game_count': 1,
                'covered_count': int(next_away_game['Spread Covered Away']),
                'Win/Loss': away_team_result
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Group by 'Gap (Days)', 'Team Type', 'Win/Loss' to get counts and coverage percentages
final_results = results_df.groupby(['Gap (Days)', 'Team Type', 'Win/Loss']).agg(
    game_count=('game_count', 'sum'),
    covered_count=('covered_count', 'sum')
).reset_index()

final_results['Percentage Covered'] = (final_results['covered_count'] / final_results['game_count']) * 100
final_results = final_results[final_results['game_count']> 20]

# Filter to only include gaps of interest
day_gaps_of_interest = [6, 7, 8, 10, 14]
gap_final_results = final_results[final_results['Gap (Days)'].isin(day_gaps_of_interest)]
gap_final_results = gap_final_results
#print(gap_final_results)

# Perform binomial test for statistical significance (p-value < 0.05)
results_with_significance = []
for index, row in final_results.iterrows():
    p_value = binom.cdf(row['covered_count'], row['game_count'], 0.5)  # Using cdf for two-sided test
    if p_value < 0.2:
        results_with_significance.append(row)

# Convert significant results to DataFrame
gaps_ = pd.DataFrame(results_with_significance)


# In[5]:


### OVERTIME

df = pd.read_excel(file_path)
df = df.sort_values(by='Date')

# List to store results
results = []

# Loop through each game
for index, row in df.iterrows():
    if row['Overtime?'] == 'Y':
        game_date = row['Date']
        home_team = row['Home Team']
        away_team = row['Away Team']

        # Find the next game for the home team (either as home or away)
        next_home_game_home = (
            df[(df['Home Team'] == home_team) & (df['Date'] > game_date)]
            .sort_values(by='Date')
            .iloc[0] if not df[(df['Home Team'] == home_team) & (df['Date'] > game_date)].empty else None
        )
        next_home_game_away = (
            df[(df['Away Team'] == home_team) & (df['Date'] > game_date)]
            .sort_values(by='Date')
            .iloc[0] if not df[(df['Away Team'] == home_team) & (df['Date'] > game_date)].empty else None
        )

        # Determine the earliest next game for the home team
        if next_home_game_home is not None and next_home_game_away is not None:
            next_home_game = next_home_game_home if next_home_game_home['Date'] < next_home_game_away['Date'] else next_home_game_away
        elif next_home_game_home is not None:
            next_home_game = next_home_game_home
        elif next_home_game_away is not None:
            next_home_game = next_home_game_away
        else:
            next_home_game = None

        # Find the next game for the away team (either as home or away)
        next_away_game_home = (
            df[(df['Home Team'] == away_team) & (df['Date'] > game_date)]
            .sort_values(by='Date')
            .iloc[0] if not df[(df['Home Team'] == away_team) & (df['Date'] > game_date)].empty else None
        )
        next_away_game_away = (
            df[(df['Away Team'] == away_team) & (df['Date'] > game_date)]
            .sort_values(by='Date')
            .iloc[0] if not df[(df['Away Team'] == away_team) & (df['Date'] > game_date)].empty else None
        )

        # Determine the earliest next game for the away team
        if next_away_game_home is not None and next_away_game_away is not None:
            next_away_game = next_away_game_home if next_away_game_home['Date'] < next_away_game_away['Date'] else next_away_game_away
        elif next_away_game_home is not None:
            next_away_game = next_away_game_home
        elif next_away_game_away is not None:
            next_away_game = next_away_game_away
        else:
            next_away_game = None

        # Calculate the gap in days and add to the results
        if next_home_game is not None:
            gap_days_home = (next_home_game['Date'] - game_date).days
            if next_home_game['Home Team'] == home_team:
                # Home team playing at home next
                results.append({
                    'Gap (Days)': gap_days_home,
                    'Team Type': 'Home-H to H',
                    'game_count': 1,
                    'covered_count': int(next_home_game['Spread Covered Home'])
                })
            else:
                # Home team playing away next
                results.append({
                    'Gap (Days)': gap_days_home,
                    'Team Type': 'Home-H to A',
                    'game_count': 1,
                    'covered_count': int(next_home_game['Spread Covered Away'])
                })

        if next_away_game is not None:
            gap_days_away = (next_away_game['Date'] - game_date).days
            if next_away_game['Home Team'] == away_team:
                # Away team playing at home next
                results.append({
                    'Gap (Days)': gap_days_away,
                    'Team Type': 'Away-A to H',
                    'game_count': 1,
                    'covered_count': int(next_away_game['Spread Covered Home'])
                })
            else:
                # Away team playing away next
                results.append({
                    'Gap (Days)': gap_days_away,
                    'Team Type': 'Away-A to A',
                    'game_count': 1,
                    'covered_count': int(next_away_game['Spread Covered Away'])
                })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Group by 'Gap (Days)' and 'Team Type' to get counts and coverage percentages
final_results = results_df.groupby(['Gap (Days)', 'Team Type']).agg(
    game_count=('game_count', 'sum'),
    covered_count=('covered_count', 'sum')
).reset_index()

final_results['Percentage Covered'] = (final_results['covered_count'] / final_results['game_count']) * 100


# Filter to only include gaps of interest
day_gaps_of_interest = [6, 7, 8, 10, 14]
final_results = final_results[final_results['Gap (Days)'].isin(day_gaps_of_interest)]
overtime_final_results = final_results[final_results['game_count'] > 10]
#print(final_results)
# Perform binomial test for statistical significance (p-value < 0.05)
results_with_significance = []
for index, row in final_results.iterrows():
    # Perform binomial test: p=0.5 for 50% coverage (null hypothesis)
    p_value = binom.cdf(row['covered_count'], row['game_count'], 0.5)  # Using cdf for two-sided test
    if p_value < 0.05:
        results_with_significance.append(row)

# Convert significant results to DataFrame
overtime_sig = pd.DataFrame(results_with_significance)


# In[6]:


### SPREAD SUCCESS

df = pd.read_excel(file_path)  # Update with your actual data path

# Define all possible spreads (e.g., from -10 to +10 in 0.5 increments)
spreads = [x / 2 for x in range(-42, 42)]  # This gives spreads from -10.0 to 10.0 in 0.5 increments

# List to store betting results
bet_results = []

# Loop through each game
for index, row in df.iterrows():
    home_team = row['Home Team']
    away_team = row['Away Team']
    home_score = row['Home Score']
    away_score = row['Away Score']
    spread_home = row['Home Line Close']
    spread_away = row['Away Line Close']

    # Home team needs to win by more than the spread
    if home_score + spread_home > away_score:
        bet_success_home = 1
    elif home_score + spread_home < away_score:
        bet_success_home = 0
    else:
        bet_success_home = None  # Exclude pushes by setting it to None

    if bet_success_home is not None:  # Only append non-push results
        bet_results.append({
            'Home Team': home_team,
            'Away Team': away_team,
            'Spread': spread_home,
            'Bet Success Home': bet_success_home
        })

# Convert betting results to DataFrame
bet_results_df = pd.DataFrame(bet_results)

# Group by 'Spread' and calculate sum and count for 'Bet Success Home'
spread_success = bet_results_df.groupby('Spread')[['Bet Success Home']].agg(['sum', 'count'])

# Calculate the cover percentage
spread_success[('Cover %')] = round(
    (spread_success[('Bet Success Home', 'sum')] / spread_success[('Bet Success Home', 'count')]) * 100,
    ndigits=0
)

# Now apply the filter to only include spreads with at least 10 games
spreads = spread_success[(spread_success[('Bet Success Home', 'count')] >= 10)]

# Display the results
#print(spreads)


# In[7]:


### DOME TEAMS

dome_teams = ['New Orleans Saints', 'Detroit Lions', 'Atlanta Falcons','Dallas Cowboys', 'Indianapolis Colts', 'Houston Texans']
cold_teams = ['Green Bay Packers', 'Kansas City Chiefs', 'Chicago Bears', 'Buffalo Bills', 'New England Patriots', 'Baltimore Ravens', 'Pittsburgh Steelers']
hot_teams = ['Arizona Cardinals', 'Jacksonville Jaguars', 'Las Vegas Raiders', 'Los Angeles Rams', 'Los Angeles Chargers', 'Miami Dolphins']

df = pd.read_excel(file_path)
df = df[df['Push'] == 1]

def dome_teams_away():
    # Filter for dome teams vs cold teams
    filtered_df = df[(df['Away Team'].isin(dome_teams)) & (df['Home Team'].isin(cold_teams))]
    hot_df = df[(df['Away Team'].isin(hot_teams)) & (df['Home Team'].isin(cold_teams))]

    # Calculate coverage for cold teams vs dome teams
    cold_teams_cover_dome = filtered_df['Spread Covered Home'].mean()  # Coverage when cold team is at home
    dome_teams_cover_cold = filtered_df['Spread Covered Away'].mean()  # Coverage when dome team is away
    
    # Group by Home Team and aggregate
    filtered_df_grouped = filtered_df.groupby('Home Team').agg({
        'Spread Covered Home': ['sum', 'count', 'mean']
    })
      
    hot_teams_cover_cold = hot_df['Spread Covered Away'].mean()  # Coverage when hot team is away at cold team
    cold_teams_cover_hot = hot_df['Spread Covered Home'].mean()  # Coverage when cold team is home against hot team
    
    # Group by Away Team and aggregate for hot teams
    hot_df_grouped = hot_df.groupby('Away Team').agg({
        'Spread Covered Away': ['sum', 'count', 'mean']
    })
    
    
    return filtered_df, cold_teams_cover_dome, dome_teams_cover_cold, hot_teams_cover_cold, cold_teams_cover_hot

# Call the function
filtered_df, cold_teams_cover_dome, dome_teams_cover_cold, hot_teams_cover_cold, cold_teams_cover_hot = dome_teams_away()

# Print the results

print(f"Cold Teams Cover Dome @ {cold_teams_cover_dome:.2f}%")
print(f"Dome Teams Cover Cold @ {dome_teams_cover_cold:.2f}%")
print(f"Cold Teams Cover Hot @ {cold_teams_cover_hot:.2f}%")
print(f"Hot Teams Cover Cold @ {hot_teams_cover_cold:.2f}%")



# In[8]:


### Previous Points Totals

df = pd.read_excel(file_path,sheet_name='Linear')  # Update with your actual data path

df = df.sort_values(by='Date',ascending=True)

###df = df[(df['Date'] >= '2024-10-01')]

def predict_next_score():
    # Load data
        
    # Combine scores for simplicity, as one option
    df['Previous_5_Scores'] = df.groupby('Team')['Score'].shift().rolling(window=5).mean()
    df['Previous_10_Scores'] = df.groupby('Team')['Score'].shift().rolling(window=10).mean()
    df['Previous_20_Scores'] = df.groupby('Team')['Score'].shift().rolling(window=20).mean()
    df['Previous_40_Scores'] = df.groupby('Team')['Score'].shift().rolling(window=40).mean()
    df['Previous_80_Scores'] = df.groupby('Team')['Score'].shift().rolling(window=80).mean()
    df['Next Game Score'] = df.groupby('Team')['Score'].shift(-1).rolling(window=1).mean()

    df_5 = df.dropna(subset=['Previous_5_Scores', 'Next Game Score'])
    df_10 = df.dropna(subset=['Previous_10_Scores', 'Next Game Score'])
    df_20 = df.dropna(subset=['Previous_20_Scores', 'Next Game Score'])
    df_40 = df.dropna(subset=['Previous_40_Scores', 'Next Game Score'])
    df_80 = df.dropna(subset=['Previous_80_Scores', 'Next Game Score'])
    
    df_10['10-NGS'] = df_10['Previous_10_Scores'] - df_10['Next Game Score']
    df_5['5-NGS'] = df_5['Previous_5_Scores'] - df_5['Next Game Score']
    df_20['20-NGS'] = df_20['Previous_20_Scores'] - df_20['Next Game Score']
    df_40['40-NGS'] = df_40['Previous_40_Scores'] - df_40['Next Game Score']
    df_80['80-NGS'] = df_80['Previous_80_Scores'] - df_80['Next Game Score']

    
    #print("5 Games",df_5['5-NGS'].mean(),df_5['5-NGS'].std(),df_5['5-NGS'].var())
    #print("10 Games",df_10['10-NGS'].mean(),df_10['10-NGS'].std(),df_10['10-NGS'].var())
    #print("20 Games",df_20['20-NGS'].mean(),df_20['20-NGS'].std(),df_20['20-NGS'].var())
    #print("40 Games",df_40['40-NGS'].mean(),df_40['40-NGS'].std(),df_40['40-NGS'].var())
    #print("80 Games",df_80['80-NGS'].mean(),df_80['80-NGS'].std(),df_80['80-NGS'].var())

    
    return df[['Date','Team','Previous_5_Scores','Previous_10_Scores','Next Game Score']]

#df[['Date', 'Team','Previous_5_Scores','Previous_10_Scores','Next Game Score']]= predict_next_score() 
#print(df[['Date', 'Team','Previous_5_Scores','Previous_10_Scores','Next Game Score']])


# In[10]:


### Locations & Teams

titans = ["Tennessee Titans",36.166461,-86.771289]
giants = ["New York Giants", 40.812194,-74.076983]	
steelers = ["Pittsburgh Steelers",40.446786,-80.015761]
panthers = ["Carolina Panthers",35.225808,-80.852861]
ravens = ["Baltimore Ravens",39.277969,-76.622767]
buccaneers = ["Tampa Bay Buccaneers",27.975967,-82.50335]
colts = ["Indianapolis Colts",39.760056,-86.163806]
vikings	= ["Minnesota Vikings",44.973881,-93.258094]
cardinals = ["Arizona Cardinals", 33.5277,-112.262608]
cowboys	= ["Dallas Cowboys", 32.747778,-97.092778]
falcons	= ["Atlanta Falcons", 33.757614,-84.400972]
jets = ["New York Jets", 40.812194,-74.076983]
broncos	= ["Denver Broncos",39.743936,-105.020097]
dolphins = ["Miami Dolphins", 25.957919,-80.238842]
eagles = ["Philadelphia Eagles",39.900775,-75.167453]
bears = ["Chicago Bears",41.862306,-87.616672]
patriots = ["New England Patriots", 42.090925,-71.26435]
commanders = ["Washington Commanders",38.907697,-76.864517]
packers	= ["Green Bay Packers", 44.501306,-88.062167]
chargers = ["Los Angeles Chargers",32.783117,-117.119525]
saints = ["New Orleans Saints",29.950931,-90.081364]
texans = ["Houston Texans",29.684781,-95.410956]
bills = ["Buffalo Bills",42.773739,-78.786978]
f9ers = ["San Francisco 49ers",37.713486,-122.386256]
jaguars	= ["Jacksonville Jaguars",30.323925   ,-81.637356]
browns = ["Cleveland Browns",41.506022	,-81.699564]
raiders	= ["Las Vegas Raiders",37.751411	,-122.200889]
chiefs = ["Kansas City Chiefs",39.048914	,-94.484039]
rams = ["Los Angeles Rams",38.632975	,-118.188547]
seahawks = ["Seattle Seahawks",47.595153	,-122.331625]
bengals	= ["Cincinnati Bengals",39.095442	,-84.516039]
lions = ["Detroit Lions",42.340156	,-83.045808]

teams = [titans,giants,steelers,panthers,ravens,buccaneers,colts,vikings,cardinals,cowboys,falcons,
         jets,broncos,dolphins,eagles,bears,patriots,commanders,packers,chargers,saints,texans,bills,f9ers,
         jaguars,browns,raiders,chiefs,rams,seahawks,bengals,lions]


# In[11]:


dist = []
k = 0

for i in range(0,len(teams)-1):
    lat1,lon1 = map(radians,[teams[i][1],teams[i][2]])
    #print(teams[i],"team")
    for j in range(k,len(teams)-1):
        
        lat2,lon2 = map(radians,[teams[j+1][1],teams[j+1][2]])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        # Radius of Earth in kilometers (use 3956 for miles)
        r = 6371.0
        distance = r * c
        dist.append((teams[i][0],teams[j+1][0],distance))
    k = k + 1



df = pd.read_excel(file_path,sheet_name='Data')
df = df.sort_values(by='Home Team',ascending=False)

away_travel = []

for i in range(0,len(df['Home Team'])):
    for j in range(0,len(dist)):
        if (df['Home Team'][i] == dist[j][0] or df['Home Team'][i] == dist[j][1]) and (df['Away Team'][i] == dist[j][0] or df['Away Team'][i] == dist[j][1]):
            away_travel.append((df['Home Team'][i],df['Away Team'][i],dist[j][2]))
            break

df['Distance'] = np.nan  # Initialize the 'Distance' column with NaN

# Iterate through the rows of the DataFrame
def distance():
    for i, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        
        # Find the corresponding distance in the array
        for item in away_travel:
            if item[0] == home_team and item[1] == away_team:
                df.at[i, 'Distance'] = item[2]  # Set the distance in the DataFrame

    # Make sure 'Distance' column is numeric and handle missing values if needed
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')  # Converts to NaN if not numeric

    #print(df.shape[0])
    # Drop rows where 'Distance' or 'Spread Covered Home' is NaN
    df_cleaned = df.dropna(subset=['Distance', 'Spread Covered Home'])

    # Count all unique values
    counts_all = df_cleaned['Distance'].value_counts()
    #print(counts_all,"counts")
    home_cover = (df_cleaned.groupby('Distance')['Spread Covered Home'].sum())*100 / df_cleaned.groupby('Distance')['Spread Covered Home'].count()

    # Filter the original DataFrame to only include rows with the selected distances
    # Map the calculated Home Cover back to all rows in the DataFrame
    df_cleaned['Home Cover'] = df_cleaned['Distance'].map(home_cover)
    #print(df_cleaned.shape[0],"lool")
    team_distances = df_cleaned
    df_distance = df_cleaned[df_cleaned['Distance'].isin(df_cleaned.groupby('Distance')['Spread Covered Home'].sum()[df_cleaned.groupby('Distance')['Spread Covered Home'].sum() > 5].index)]
    #print(df_distance.shape[0],"lol")
    correlationdistance = df_distance['Distance'].corr(df_distance['Home Cover'])

    df_distance['Distance Count'] = df_distance.groupby('Distance')['Distance'].transform('count')
    
    # Prepare the data for weighted regression
    X = df_distance['Distance']  # Independent variable (Distance)
    y = df_distance['Home Cover']  # Dependent variable (Home Cover %)
    weights = df_distance['Distance Count']  # Weights based on the count of each distance
    
    # Add a constant to the model (intercept term)
    X = sm.add_constant(X)
    
    # Fit the weighted linear regression model
    model = sm.WLS(y, X, weights=weights)
    results = model.fit()
    
    # Get the predicted values (Home Cover % based on Distance)
    df_distance['Predicted Home Cover'] = results.predict(X)
    df_distance = df_distance.sort_values(by='Distance')
    
    # Display the model summary and the predicted Home Cover
    #print(results.summary())
    
    # Return the relevant columns for further analysis
    return df_distance[['Home Team', 'Away Team', 'Distance','Home Cover','Predicted Home Cover']], correlationdistance, team_distances

# Print the updated DataFrame
df_distance, correlationdistance, team_distances = distance()
team_distances = team_distances
print(team_distances,"team distances")
df_distance = df_distance
print(df_distance.shape[0])
df_distance_unique = df_distance.drop_duplicates(subset=['Home Team', 'Away Team'])
print(df_distance_unique, df_distance_unique.shape[0])
correlationdistance = correlationdistance
print(correlationdistance)
#print(f"Correlation between Away distance and Home Cover {round(correlation,ndigits=2)}")
