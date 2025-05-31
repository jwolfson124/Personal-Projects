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
