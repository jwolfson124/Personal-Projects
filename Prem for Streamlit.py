#!/usr/bin/env python
# coding: utf-8

# In[963]:


#this cell once everything is run will hold all of the final datasets
#display(prem_table.head(2)) #this is the premier league table for the last 3 years with points, goal difference, and goals for
#display(most_competitive_teams.head(2)) ##which teams were the hardest to beat home and away
#display(hardest_stadiums_win.head(2))##Hardest to beat at home


# In[61]:


#import libraries and packages
import kagglehub
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import PercentFormatter
import numpy as np
import streamlit as st
import matplotlib as plt
import altair as alt
st.set_page_config(layout='wide') #make sure we can use the entire streamlit page


# In[63]:


#take the path from the kaggle website
path = kagglehub.dataset_download("evangower/premier-league-matches-19922022")
os.listdir(path)
df = pd.read_csv(os.path.join(path, 'premier-league-matches.csv'))
df.head(3)

#make the sure the dataset is only looking at the last 3 years of premier league
df = df[df['Season_End_Year'] >= 2021]

#df


# # Dataset Updates

# ### add in the stadium name for the teams

# In[65]:


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


# ### add in the name of the stadium and the capacity for the home team

# In[67]:


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


# In[181]:


#display(df.head(5))


# ## Add Columns to the DF

# In[69]:


#using the where functioon, which operates as in if statement add in the winning team column
df['Winning_Team'] = np.where(df['FTR'] == 'A', df['Away'], np.where(df['FTR'] == 'H', df['Home'], 'D'))
df['points'] = np.where(df['Winning_Team'] != 'D', 3, 1)

df['Goal Difference'] = abs(df['HomeGoals'] - df['AwayGoals'])


# # Building the Premier League Table

# In[71]:


##split the df into winning teams vs draws. Use this to properly give points to the teams
tie_df = df[df['Winning_Team'] == 'D']

#Home teams points then away teams points
#columns_to_group = ['Season_End_Year', 'Home']


home_results = tie_df.groupby(['Season_End_Year', 'Home'], as_index=False)['points'].sum().rename(columns={'Home':'Team'})
away_results = tie_df.groupby(['Season_End_Year', 'Away'], as_index=False)['points'].sum().rename(columns={'Away':'Team'})

draw_points = pd.concat([home_results,away_results]).groupby(['Season_End_Year', 'Team'], as_index=False)['points'].sum()


# In[ ]:





# In[73]:


#create a df that holds all points for winning teams then join to draw_points to recreate the prem table over the last 3 years
#remove all instances of D

#create the path
win_results = df[df['FTR'] != 'D'].copy()

win_points = win_results.groupby(['Season_End_Year', 'Winning_Team'], as_index=False)['points'].sum()
win_points = win_points.rename(columns={'Winning_Team' : 'Team'})


#display(win_points.head())
#display(draw_points.head())

#merge and group by then sort to get the final premiere league table
points_df = pd.concat([win_points,draw_points])
points_tally_df = points_df.groupby(['Season_End_Year', 'Team'], as_index=False)['points'].sum()

final_table = points_tally_df.sort_values(['Season_End_Year', 'points'], ascending=[True,False]).reset_index(drop=True)

#final_table


# In[75]:


##calculate the goals scored
home_goals = df[['Season_End_Year','Home', 'HomeGoals']].copy()
home_goals = home_goals.groupby(['Season_End_Year', 'Home'], as_index=False)['HomeGoals'].sum().rename(columns={'Home' : 'Team', 'HomeGoals': 'Goals'})

away_goals = df[['Season_End_Year','Away', 'AwayGoals']].copy()
away_goals = away_goals.groupby(['Season_End_Year', 'Away'], as_index=False)['AwayGoals'].sum().rename(columns={'Away' : 'Team', 'AwayGoals': 'Goals'})

goals_scored = pd.concat([home_goals, away_goals]).groupby(['Season_End_Year', 'Team'], as_index=False)['Goals'].sum()

#merge the datasets with the goals
points_goals_df = pd.merge(final_table, goals_scored, on=['Season_End_Year','Team'])


# In[77]:


#groupy by the home then away team and sum the different goals
home = df.groupby(['Season_End_Year', 'Home'], as_index=False)[['HomeGoals', 'AwayGoals']].sum().rename(columns={'Home': 'Team'})
away = df.groupby(['Season_End_Year', 'Away'], as_index=False)[['HomeGoals', 'AwayGoals']].sum().rename(columns={'Away': 'Team'})

#calculate the goal difference home and away
home['Goal Difference'] = home['HomeGoals'] - home['AwayGoals']
away['Goal Difference'] = away['AwayGoals'] - away['HomeGoals']

goal_difference = pd.concat([home, away]).groupby(['Season_End_Year', 'Team'], as_index=False)['Goal Difference'].sum()

#join the final prem teams
final_prem_table = pd.merge(points_goals_df, goal_difference, on=['Season_End_Year', 'Team'])


# In[79]:


prem_table = final_prem_table.sort_values(['Season_End_Year','points', 'Goal Difference', 'Goals'], ascending=[True, False, False, False])

prem_table['rank'] = prem_table.groupby('Season_End_Year').cumcount() + 1

#create the final premiere league table
prem_table = prem_table[['Season_End_Year', 'Team', 'rank', 'points', 'Goal Difference', 'Goals']]

#prem_table


# # End of Premier League Table Build

# # The Teams that are hardest to beat Home and Away

# In[81]:


#take the tie dataframe that we had before and then also create a dataframe that holds not ties
#tie_df
not_tie_df = df[df['FTR'] != 'D']

#for all games that are not ties count the number of wins for the winning team
win_count = not_tie_df.groupby(['Season_End_Year', 'Winning_Team'], as_index=False)['FTR'] \
.count().rename(columns={'Winning_Team' : 'Team','FTR' : 'Games'})

#count the number of ties there are for the 
tie_home = tie_df.groupby(['Season_End_Year', 'Home'], as_index=False)['FTR'].count().rename(columns={'Home' : 'Team'})
tie_away = tie_df.groupby(['Season_End_Year', 'Away'],as_index=False)['FTR'].count().rename(columns={'Away' : 'Team'})

#merge the two dataframes together and then sum FTR_x and FTR_y to create the total ties column
tie_games = pd.merge(tie_home, tie_away, on=['Season_End_Year','Team'])
tie_games['Games'] = tie_games['FTR_x'] + tie_games['FTR_y']

tie_games = tie_games.drop(['FTR_x', 'FTR_y'], axis=1)

#merge the wins with the tie games
all_games = pd.concat([tie_games, win_count]).groupby(['Season_End_Year', 'Team'], as_index=False)['Games'].sum()

#organize the teams and then rank them by most competitive
comp_teams = all_games.sort_values(by=['Season_End_Year', 'Games'], ascending=[False, False])
comp_teams['rank'] = comp_teams.groupby('Season_End_Year').cumcount() + 1

#comp_teams.head(1)


# # End of the most competitive teams build

# # Hardest Stadiums To Visit

# In[83]:


away_winners = df[df['Away'] == df['Winning_Team']]
away_winners = away_winners.groupby(['Season_End_Year', 'Stadium', 'Capacity', 'Home'], as_index=False)['points'].count()\
.rename(columns={'points' : 'Games Lost', 'Home' : 'Team'})

#join in the premier league dataset and used rank to finish the sort for ties
rank_table = prem_table[['Season_End_Year', 'Team', 'rank']]
away_winners = pd.merge(away_winners, rank_table, on=['Season_End_Year', 'Team'])

#sort the dataframe by the 
hardest_stadiums_win = away_winners.sort_values(by=['Season_End_Year', 'Games Lost', 'rank'], ascending=[False, True, True])
hardest_stadiums_win['Odds of Winning'] = round(hardest_stadiums_win['Games Lost'] / 19,4)

#display(hardest_stadiums_win.head(2))


# In[85]:


#write an intro
st.title(":orange[Premier League Dashboard]")
st.write("This dashboard will give you insight into the statistics within the 2020, 2021, and 2022 Premier League Seasons")
#st.write("Please select a season to view")

#identify all available seasons and provide options to view
seasons = df['Season_End_Year'].unique()
select_season = st.selectbox("Select a Premier League Season to get the Stats!", seasons)


# ## Add Totals / Averages

# In[87]:


year_table = prem_table[prem_table['Season_End_Year'] == select_season]

#total goals + average goals
total_goals = sum(year_table['Goals'])

average_goals = round(total_goals / (38 * 10),2)

#most goals in a game
games = df[df['Season_End_Year'] == select_season].copy()
games['Total Goals'] = games['HomeGoals'] + games['AwayGoals']

most_goals_in_game = max(games['Total Goals'])

#create the columns that are going to hold the values
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    #st.subheader('Total Premier League Goals')
    st.metric(label='Total Premier League Goals',value=total_goals)

with col2:
    #st.subheader('Most Goals Scored in 1 Match')
    st.metric(label='Most Goals Scored in 1 Match', value=most_goals_in_game)

with col3:
    #st.subheader('Average Goals Per Game')
    st.metric(label='Average Goals Per Game', value=average_goals)

#most challenging stadiums to visit win percentage
stadium_visit = hardest_stadiums_win[hardest_stadiums_win['Season_End_Year']==select_season]
index_use = stadium_visit['Odds of Winning'].idxmin()

#find the odds of winning, team, and stadium
odds_of_winning = round(stadium_visit['Odds of Winning'][index_use],3)
odds_of_winning = str(odds_of_winning*100) + "%"

stadium, team = stadium_visit[['Stadium', 'Team']].loc[index_use]

with col4:
    #st.subheader(f"Odds of Winning when visiting {team}'s {stadium}")
    st.metric(label = f"Lowest Away Win Odds: {team} – {stadium}", value=odds_of_winning)


#least amount of games lost
comp_year = comp_teams[comp_teams['Season_End_Year']==select_season]
comp_index = comp_year['Games'].idxmax()

lost_games = 38 - comp_year.loc[comp_index]['Games']
team = comp_year.loc[comp_index]['Team']

with col5:
    #st.subheader(f"Least Amount of Games Lost: {team}")
    st.metric(label = f"Least Amount of Games Lost: {team}", value=lost_games)


# # Create Colored Prem Table

# In[316]:


#rename columns and filter values
#select_season=2023
filtered_prem_table = prem_table.rename(columns={'points': 'Points', 'rank' : 'Rank'})
filtered_prem_table = filtered_prem_table[prem_table['Season_End_Year'] == select_season].drop(columns = 'Season_End_Year').set_index('Rank')

def highlight_color(row):
    if row.name == 1:
        return ['background-color: khaki; color: black'] * len(row)
    elif row.name == 2:
        return ['background-color: lightgreen; color: black'] * len(row)
    elif row.name < 5:
        return ['background-color: lightblue; color: black'] * len(row)
    elif row.name < 18:
        return ['background-color: white; color: black'] * len(row)
    else:
        return ['background-color: red; color: black'] * len(row)
        

prem_table_image = filtered_prem_table.style.apply(highlight_color, axis=1)


# In[91]:


col1, col2 = st.columns(2)

with col1:
    st.subheader('League Table')
    st.dataframe(prem_table_image)


# # Goals per Team with Average Line

# In[318]:


#goals scored by rank
filtered_prem_table = filtered_prem_table.reset_index().rename(columns={'index':'Rank'})
x = filtered_prem_table['Team']
y = filtered_prem_table['Goals']
average_goals = filtered_prem_table['Goals'].mean()

#create chart 1
chart = alt.Chart(filtered_prem_table).mark_bar().encode(
    x=alt.X('Team:N', sort=filtered_prem_table['Team'].tolist(), title='Team'),
    y=alt.Y('Goals:Q', title='Goals Scored'),
    tooltip=['Rank','Team', 'Goals'],
    color=alt.Color('Goals:Q', scale=alt.Scale(scheme='blues'))
)

#create mean line
mean_line = alt.Chart(pd.DataFrame({'Average Goals': [average_goals]})).mark_rule(color='red'
                                                                     ).encode(y='Average Goals:Q')

#create final image
final_chart = (chart + mean_line).properties(
    width=600,
    height=400)

with col2:
    st.subheader('Goals by Rank')
    st.altair_chart(final_chart)

#final_chart


# ## Goals Over the Season

# In[ ]:





# In[320]:


col1, col2 = st.columns(2)


#filtered_prem_table['Points'] = pd.to_numeric(filtered_prem_table['Points'])

#goal difference vs points
scatter_plot = alt.Chart(filtered_prem_table).mark_circle(size=100).encode(
    x=alt.X('Goal Difference:Q', title = 'Goal Difference'),
    y=alt.Y('Points:Q', title='Points'),
    tooltip=['Team', 'Goal Difference', 'Points'],
    color=alt.Color('Goal Difference:Q', sort=alt.EncodingSortField(field='Rank', order='ascending'))
).properties(
    width=650,
    height=400
)

with col1:
    st.subheader('Goal Difference vs Points')
    st.altair_chart(scatter_plot)


# In[322]:


## Weekly Goals
goals_df = df[df['Season_End_Year'] == select_season].copy()
goals_df['Total Goals'] = goals_df['HomeGoals'] + goals_df['AwayGoals']

goals = pd.DataFrame(goals_df.groupby('Wk')['Total Goals'].sum())
goals = goals.reset_index()

weekly_goals = alt.Chart(goals).mark_line(point=True).encode(
    x=alt.X('Wk:O', title='Match Day'),
    y=alt.Y('Total Goals:Q', title='Weekly Goals'),
    tooltip='Total Goals'
).properties(
    width=650,
    height=400
)

with col2:
    st.subheader('Weekly Goal Trend')
    st.altair_chart(weekly_goals)



# In[ ]:




