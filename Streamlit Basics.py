#import necessary libraries
import kagglehub
import numpy as np
import pandas as pd
import os
import streamlit as st

#create the intial title
st.title("""
:orange[Premier League Dashboard]
""")
st.write("This is my :green[**FIRST**] attempt at creating a dashboard using Streamlit!")
st.write("This dashboard will contain data looking into the Premier League Seasons 2020, 2021, and 2022!")

#USE KAGGLE API TO BRING IN THE DATA
#take the path from the kaggle website
path = kagglehub.dataset_download("evangower/premier-league-matches-19922022")
os.listdir(path)
df = pd.read_csv(os.path.join(path, 'premier-league-matches.csv'))
df.head(3)

#make the sure the dataset is only looking at the last 3 years of premier league
df = df[df['Season_End_Year'] >= 2021]


#ENTER ADDITIONAL DATA TO THE DATASET AND ADDITIONAL COLUMNS
#STADIUM
team_to_stadium = {
    'Arsenal': 'Emirates Stadium',
    'Aston Villa': 'Villa Park',
    'Bournemouth': 'Vitality Stadium',
    'Brentford': 'GTech Community Stadium',
    'Brighton': 'Amex Stadium',
    'Burnley': 'Turf Moor',
    'Chelsea': 'Stamford Bridge',
    'Crystal Palace': 'Selhurst Park',
    'Everton': 'Goodison Park',
    'Fulham': 'Craven Cottage',
    'Leeds United': 'Elland Road',
    'Leicester City': 'King Power Stadium',
    'Liverpool': 'Anfield',
    'Manchester City': 'Etihad Stadium',
    'Manchester Utd': 'Old Trafford',
    'Newcastle Utd': "St James' Park",
    'Norwich City': 'Carrow Road',
    "Nott'ham Forest": 'City Ground',
    'Sheffield Utd': 'Bramall Lane',
    'Southampton': "St Mary's Stadium",
    'Tottenham': 'Tottenham Hotspur Stadium',
    'Watford': 'Vicarage Road',
    'West Brom': 'The Hawthorns',
    'West Ham': 'London Stadium',
    'Wolves': 'Molineux Stadium'
}



def get_stadium(row):
    return team_to_stadium[row['Home']]

df['Stadium'] = df.apply(get_stadium, axis=1)

#CAPACITY
##build the capacity dictionaries
capacity_2021 = {
    'Arsenal': 60704,  # Emirates Stadium
    'Aston Villa': 42682,  # Villa Park
    'Bournemouth': 11307,  # Vitality Stadium
    'Brentford': 17250,  # Brentford Community Stadium
    'Brighton': 31876,  # Amex Stadium
    'Burnley': 21944,  # Turf Moor
    'Chelsea': 40834,  # Stamford Bridge
    'Crystal Palace': 25486,  # Selhurst Park
    'Everton': 39414,  # Goodison Park
    'Fulham': 25700,  # Craven Cottage
    'Leeds United': 37890,  # Elland Road
    'Leicester City': 32312,  # King Power Stadium
    'Liverpool': 54074,  # Anfield
    'Manchester City': 55097,  # Etihad Stadium
    'Manchester Utd': 74994,  # Old Trafford
    'Newcastle Utd': 52338,  # St James' Park
    'Norwich City': 27244,  # Carrow Road
    "Nott'ham Forest": 30445,  # City Ground
    'Sheffield Utd': 32702,  # Bramall Lane
    'Southampton': 32384,  # St Mary's Stadium
    'Tottenham': 62062,  # Tottenham Hotspur Stadium
    'Watford': 22200,  # Vicarage Road
    'West Brom': 26688,  # The Hawthorns
    'West Ham': 60000,  # London Stadium
    'Wolves': 31750  # Molineux Stadium
}
capacity_2022 = {
    'Arsenal': 60704,
    'Aston Villa': 42657,
    'Bournemouth': 11307,
    'Brentford': 17250,
    'Brighton': 31800,
    'Burnley': 21944,
    'Chelsea': 40343,
    'Crystal Palace': 25486,
    'Everton': 39414,
    'Fulham': 25700,
    'Leeds United': 37892,
    'Leicester City': 32312,
    'Liverpool': 54074,
    'Manchester City': 53400,
    'Manchester Utd': 74310,
    'Newcastle Utd': 52305,
    'Norwich City': 27244,
    "Nott'ham Forest": 30445,
    'Sheffield Utd': 32702,
    'Southampton': 32384,
    'Tottenham': 62850,
    'Watford': 22200,
    'West Brom': 26688,
    'West Ham': 62500,
    'Wolves': 32050
}
capacity_2023 = {
    'Arsenal': 60704,
    'Aston Villa': 42657,
    'Bournemouth': 11307,
    'Brentford': 17250,
    'Brighton': 31800,
    'Burnley': 21944,
    'Chelsea': 40343,
    'Crystal Palace': 25486,
    'Everton': 39414,
    'Fulham': 27000,  # Riverside Stand phased expansion underway
    'Leeds United': 37892,
    'Leicester City': 32312,
    'Liverpool': 61000,  # Anfield Road Stand expansion completed late 2023
    'Manchester City': 53400,
    'Manchester Utd': 74310,
    'Newcastle Utd': 52305,
    'Norwich City': 27244,
    "Nott'ham Forest": 30445,  # Small capacity increase with pod seating
    'Sheffield Utd': 32702,
    'Southampton': 32384,
    'Tottenham': 62850,
    'Watford': 22200,
    'West Brom': 26688,
    'West Ham': 62500,
    'Wolves': 32050
}




seasons = {
    2021 : capacity_2021,
    2022 : capacity_2022,
    2023 : capacity_2023
}

row = df.iloc[0]
def get_capacity(row):
    capacity_dict = seasons[row['Season_End_Year']]
    
    return capacity_dict[row['Home']]

df['Capacity'] = df.apply(get_capacity, axis=1)

#ADD IN ADDITIONAL COLUMNS
#using the where functioon, which operates as in if statement add in the winning team column
df['Winning_Team'] = np.where(df['FTR'] == 'A', df['Away'], np.where(df['FTR'] == 'H', df['Home'], 'D'))
df['points'] = np.where(df['Winning_Team'] != 'D', 3, 1)

df['Goal Difference'] = abs(df['HomeGoals'] - df['AwayGoals'])
