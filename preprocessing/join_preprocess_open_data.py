# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:24:02 2025

This script will perform preprocessing on the Somerville open datasets and save
the resultant joined dataframes as a pkl file to be read by the app.py file.

This script is largely copy and pasted from the SomervilleDataExploration.ipynb
file within my Data Exploration repository.

@author: Kevin Monk
"""

import pandas as pd
import numpy as np
import scipy
import sklearn
import os
import matplotlib.pyplot as plt

# Loading in data
path2csv = 'C:/Users/kevjm/Documents/GitHub/SomervilleOpenDataExploration/data'

d = os.listdir(path2csv)

df_bike_ped_counts = pd.read_csv(path2csv + '/' + d[0])
df_crashes = pd.read_csv(path2csv + '/' + d[1])
df_crime = pd.read_csv(path2csv + '/' + d[2])
df_happiness = pd.read_csv(path2csv + '/' + d[3],low_memory=False)

#%% Processing and Merging Crime and Crash Statistics:
# finding number of crimes reported per year
crime_count_yr = (df_crime
                  .groupby('Year Reported').size()
                  .to_frame(name='CrimeCount'))

# creating new Year column for crash database from Date and TIme of Crash column
df_crashes['Year'] = (pd.to_datetime(df_crashes['Date and Time of Crash'], format = 'mixed')
                      .dt.year)
crash_count_yr = (df_crashes
                  .groupby('Year').size()
                 .to_frame(name='CrashCount'))

# combine crash and crime counts
crash_crime_counts = pd.merge(crash_count_yr,crime_count_yr,
                           how='outer',
                           left_index = True, right_index = True)

# joining counts to df_happiness on the year value
df_happiness_merge = (
    pd.merge(
        df_happiness,crash_crime_counts,
        how = 'left',left_on = 'Year', 
        right_index = True
    )
)

# incorporating the number of crimes and crashes reported in each ward in each year
crime_count_yr_ward = (df_crime.
                       groupby(['Ward','Year Reported']).size()
                       .to_frame(name='CrimeCount_PerYearPerWard')
                       .reset_index()
                      )
crash_count_yr_ward = (df_crashes.
                       groupby(['Ward','Year']).size()
                       .to_frame(name='CrashCount_PerYearPerWard')
                       .reset_index()
                      )

# prepping crime data to get wards as floats for future joining:
crime_count_yr_ward['Ward'] = crime_count_yr_ward['Ward'].str.get(0) #currently a string starting with the number and followed by white space
crime_count_yr_ward = crime_count_yr_ward.loc[crime_count_yr_ward['Ward']!='C',:] #a C appeared in the Ward designation
crime_count_yr_ward['Ward'] = crime_count_yr_ward['Ward'].astype(float) #changing to float for future merges

# prepping crash data for future merges
crash_count_yr_ward['Year'] = crash_count_yr_ward['Year'].astype('int64')

# merge crash and crime statistics into single dataframe
crash_crime_yr_ward = (pd
                       .merge(crash_count_yr_ward,crime_count_yr_ward,
                              how = 'right',
                              right_on = ['Ward','Year Reported'],
                              left_on = ['Ward','Year'])
                       .drop('Year',axis=1)
)

# combine the crash and crime statistics per ward with the survey statistics by merging across multiple columns
df_happiness_merge = (pd
                     .merge(df_happiness_merge,crash_crime_yr_ward,
                           how = 'left',
                           left_on = ['Year','Ward'],
                           right_on = ['Year Reported','Ward'])
                      .drop('Year Reported',axis=1)
                     )

#%% Prepping and Merging Cyclist and Pedestrian Counts:
#group bike and pedestrian counts by year and mode (collapsing across locations and time of day)
df_grouped = (df_bike_ped_counts
              .groupby(['Year','Mode'])
              .agg({'Count': 'mean'})
              .reset_index())

# average the bike and pedestrian counts
mn_count_yr_bike = (df_grouped
                    .loc[df_grouped['Mode']=='Bike',['Year','Count']]
                    .rename(columns={'Count': 'bike_count'})
                   )
mn_count_yr_ped = (df_grouped
                    .loc[df_grouped['Mode']=='Ped',['Year','Count']]
                    .rename(columns={'Count': 'ped_count'})
                   )

# merge both counts into single data frame
mn_count_yr_merge = pd.merge(mn_count_yr_bike,mn_count_yr_ped,
                          on = 'Year',
                          how = 'left'
                         )

# creating ratio metric for pedestrian to cyclists
mn_count_yr_merge['ped2bike_foldinc'] = (
    mn_count_yr_merge['ped_count'].div(mn_count_yr_merge['bike_count'])
)

# merging counts and fold increase to happiness table
df_happiness_merge = (
    pd.merge(
        df_happiness_merge,mn_count_yr_merge,
        on = 'Year',
        how = 'left',
    )
)

#%% Preparing Dataframe for Future Modeling
# Curating rows that go into dataframe
# remove rows without a happiness score
df_happiness_merge.dropna(subset = ['Happiness.5pt.num'],inplace = True)

# identify numeric and categorical columns:
col_numeric = df_happiness_merge.select_dtypes(include=['float64','int64']).columns.tolist()
col_categorical = df_happiness_merge.select_dtypes(exclude=['float64','int64']).columns.tolist()

# find categorical columns that are not repeated within the numerical columns
col_categorical_unique = []
for s_cat in col_categorical:
    if (
        not any(s_cat in s_num for s_num in col_numeric) 
        and 'label' not in s_cat
       ):
        col_categorical_unique.append(s_cat)

# combining column lists for future modeling
col_for_mdl = col_numeric + col_categorical_unique
df_happiness_subset = df_happiness_merge.loc[:,col_for_mdl]
df_happiness_subset = df_happiness_subset.drop(['ACS.Year'],axis=1) #removing a redundant numerical column

# Imputing missing values
# Remove rows without Happiness Score
df_happiness_subset = df_happiness_subset.dropna(subset = ['Happiness.5pt.num'])

# Fill missing Ward data with a value of 8
df_happiness_subset['Ward'] = df_happiness_subset['Ward'].fillna(8)

# to fill the remaining columns with the most common value, I'll create a custom function and apply to the dataframe
def fill_w_mode(series):
    ''' this function will replace all missing values with the series-wide mode.
        Should work with numeric and cateogrical columns
    '''
    return series.fillna(series.mode().values[0])

df_happiness_prepped = df_happiness_subset.apply(fill_w_mode)
#%% Saving Pre-Processed Dataframe as pkl:
path2pkl = 'C:/Users/kevjm/Documents/GitHub/SomervilleDataStreamlit/data'
file_path = os.path.join(path2pkl,'df_happiness_prepped.pkl')
df_happiness_prepped.to_pickle(file_path)


