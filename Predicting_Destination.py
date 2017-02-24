
# coding: utf-8

# # Predicting Destination - AirBnB Customers

# The goal for this analysis is to build a neural network, using Tensorflow, that can predict which country a new user of AirBnB will make his/her first trip to. More information about the data can be found at: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings. This data was originally from a 2016 Kaggle competition, sponsored by AirBnB. I believe that it is still an excellent learning reasource because of the opportunity to clean data, engineering features, and build a model, which will all significantly impact the quality of the predictions.

# In[ ]:

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from scipy import stats
pd.set_option("display.max_columns", 1000)
import math
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import classification_report


# In[ ]:

# Load the data
countries = pd.read_csv("countries.csv")
test = pd.read_csv("test_users.csv")
train = pd.read_csv("train_users_2.csv")
sessions = pd.read_csv("sessions.csv")


# First, let's have a look at the data we are working with.

# In[ ]:

test.head()


# In[ ]:

train.head()


# In[ ]:

sessions.head(10)


# In[ ]:

print(test.shape)
print(train.shape)
print(sessions.shape)


# Our target feature is 'country_destination,' which can be found in the 'train' dataframe. Given this, let's first explore the sessions data, then merge it with the train dataframe (on 'user_id'), after we are done transforming it.

# ## Sessions

# In[ ]:

sessions.isnull().sum()


# In[ ]:

#Drop rows where user_id is null because we want to tie everything back to a user.
sessions = sessions[sessions.user_id.notnull()]


# In[ ]:

sessions.isnull().sum()


# In[ ]:

# How do nulls in action relate to action_type
sessions[sessions.action.isnull()].action_type.value_counts()


# In[ ]:

# Every action with a null value, has action_type equal to 'message_post'.
# Let's change all the null values to 'message'
sessions.loc[sessions.action.isnull(), 'action'] = 'message'


# In[ ]:

sessions.isnull().sum()


# In[ ]:

# action_type and action_detail are missing values in the same rows, this simplifies things a little.
print(sessions[sessions.action_type.isnull()].action.value_counts())
print()
print(sessions[sessions.action_detail.isnull()].action.value_counts())


# To fill in the null values for action_type and action_detail, we will perform these steps:
# 
# 1. Use the most common value relative to each user and action
# 2. Use the most common value relative to each action
# 3. Use the value 'missing'

# In[ ]:

# function that finds the most common value of a feature, specific to each user and action.
def most_common_value_by_user(merge_df, feature): 
    # Find the value counts for a feature, for each user and action.
    new_df = pd.DataFrame(merge_df.groupby(['user_id','action'])[feature].value_counts())
    # Set the index to a new feature so that it can be transformed.
    new_df['index_tuple'] = new_df.index 
    # Change the feature name to count, since it is the value count of the feature.
    new_df['count'] = new_df[feature]
    
    new_columns = ['user_id','action',feature]
    # separate the features of index_tuple (a list), into their own columns
    for n,col in enumerate(new_columns):
        new_df[col] = new_df.index_tuple.apply(lambda index_tuple: index_tuple[n])
    
    # reset index and drop index_tuple
    new_df = new_df.reset_index(drop = True)
    new_df = new_df.drop(['index_tuple'], axis = 1) 
    
    # Create a new dataframe for each user, action, and the count of the most common feature
    new_df_max = pd.DataFrame(new_df.groupby(['user_id','action'], as_index = False)['count'].max())
    # Merge dataframes to include the name of the most common feature
    new_df_max = new_df_max.merge(new_df, on = ['user_id','action','count'])
    # Drop count as it is not needed for the next step
    new_df_max = new_df_max.drop('count', axis = 1)
    
    # Merge with main dataframe (sessions)
    merge_df = merge_df.merge(new_df_max, left_on = ['user_id','action'], right_on = ['user_id','action'], how = 'left')
    
    return merge_df


# In[ ]:

sessions = most_common_value_by_user(sessions, 'action_type')
print("action_type is complete.")

sessions = most_common_value_by_user(sessions, 'action_detail')
print("action_detail is complete.")


# In[ ]:

# Replace the nulls with the values from the features created by 'most_common_value_by_user' function.
sessions.loc[sessions.action_type_x.isnull(), 'action_type_x'] = sessions.action_type_y
sessions.loc[sessions.action_detail_x.isnull(), 'action_detail_x'] = sessions.action_detail_y

# Change the features' names to their originals and drop unnecessary features.
sessions['action_type'] = sessions.action_type_x
sessions['action_detail'] = sessions.action_detail_x
sessions = sessions.drop(['action_type_x','action_type_y','action_detail_x','action_detail_y'], axis = 1)


# That helped to remove some of the nulls values. Now let's try the more general function.

# In[ ]:

sessions.isnull().sum()


# In[ ]:

# function that finds the most common value of a feature, specific to each action.
def most_common_value_by_all_users(merge_df, feature):
    # Group by action, then find the value counts of the feature
    new_df = pd.DataFrame(merge_df.groupby('action')[feature].value_counts())
    # Set the index to a new feature so that it can be transformed.
    new_df['index_tuple'] = new_df.index 
    # Change the feature name to count, since it is the value count of the feature.
    new_df['count'] = new_df[feature]
    
    new_columns = ['action',feature]
    # separate the features of index_tuple (a list), into their own columns
    for n,col in enumerate(new_columns):
        new_df[col] = new_df.index_tuple.apply(lambda index_tuple: index_tuple[n])
    
    # reset index and drop index_tuple
    new_df = new_df.reset_index(drop = True)
    new_df = new_df.drop(['index_tuple'], axis = 1) 
    
    # Create a new dataframe for each action, and the count of the most common feature
    new_df_max = pd.DataFrame(new_df.groupby('action', as_index = False)['count'].max())
    # Merge dataframes to include the name of the most common feature
    new_df_max = new_df_max.merge(new_df, on = ['action','count'])
    # Drop count as it is not needed for next step
    new_df_max = new_df_max.drop('count', axis = 1)
    
    # Merge dataframe with main dataframe (sessions)
    merge_df = merge_df.merge(new_df_max, left_on = 'action', right_on = 'action', how = 'left')
    
    return merge_df


# In[ ]:

sessions = most_common_value_by_all_users(sessions, 'action_type')
print("action_type is complete.")
sessions = most_common_value_by_all_users(sessions, 'action_detail')
print("action_detail is complete.")


# In[ ]:

# Replace the nulls with the values from the features created by 'most_common_value_by_all_users' function.
sessions.loc[sessions.action_type_x.isnull(), 'action_type_x'] = sessions.action_type_y
sessions.loc[sessions.action_detail_x.isnull(), 'action_detail_x'] = sessions.action_detail_y

# Change the features' names to their originals and drop the unnecessary features.
sessions['action_type'] = sessions.action_type_x
sessions['action_detail'] = sessions.action_detail_x
sessions = sessions.drop(['action_type_x','action_type_y','action_detail_x','action_detail_y'], axis = 1)


# There are still some null values remaining. Let's take a look at what actions these null values are related to.

# In[ ]:

sessions.isnull().sum()


# In[ ]:

sessions[sessions.action_type.isnull()].action.value_counts()


# Let's take a look at the value counts for all of the actions to see how their frequency compares to others.

# In[ ]:

sessions.action.value_counts()


# 'similar_listings_v2', 'lookup', and 'track_page_view' are the three main features with null values. I will give each of them specific values for action_type and action_detail, otherwise I will set the value to 'missing'.

# In[ ]:

# Use these values for 'similar_listings_v2' since they are similar actions.
print(sessions[sessions.action == 'similar_listings'].action_type.value_counts())
print(sessions[sessions.action == 'similar_listings'].action_detail.value_counts())


# In[ ]:

sessions.loc[sessions.action == 'similar_listings_v2', 'action_type'] = "data"
sessions.loc[sessions.action == 'similar_listings_v2', 'action_detail'] = "similar_listings"

# No other action is similar, so we'll use the same work for all three features.
sessions.loc[sessions.action == 'lookup', 'action_type'] = "lookup"
sessions.loc[sessions.action == 'lookup', 'action_detail'] = "lookup"

sessions.loc[sessions.action == 'track_page_view', 'action_type'] = "track_page_view"
sessions.loc[sessions.action == 'track_page_view', 'action_detail'] = "track_page_view"

sessions.action_type = sessions.action_type.fillna("missing")
sessions.action_detail = sessions.action_detail.fillna("missing")


# All good. Now just secs_elapsed is left.

# In[ ]:

sessions.isnull().sum()


# To keep things simple, let's fill the nulls with the median value for each action.

# In[ ]:

# Find the median secs_elapsed for each action
median_duration = pd.DataFrame(sessions.groupby('action', as_index = False)['secs_elapsed'].median())
median_duration.head()


# In[ ]:

# Merge dataframes on action
sessions = sessions.merge(median_duration, left_on = 'action', right_on = 'action', how = 'left')
print("Merge complete.")
# if secs_elapsed is null, fill it with the median value
sessions.loc[sessions.secs_elapsed_x.isnull(), 'secs_elapsed_x'] = sessions.secs_elapsed_y
print("Nulls are filled.")
# Change column name
sessions['secs_elapsed'] = sessions.secs_elapsed_x
print("Column is created.")
# Drop unneeded columns
sessions = sessions.drop(['secs_elapsed_x','secs_elapsed_y'], axis = 1)
print("Columns are dropped.")


# All clean!

# In[ ]:

sessions.isnull().sum()


# I think the best next step would be to take the information from sessions and summarize it. We will create a new dataframe, add the most important features, then join it with the train dataframe.

# ## Sessions' Summary

# In[ ]:

sessions.head()


# In[ ]:

# sessions_summary is set to the number of times a user_id appears in sessions
sessions_summary = pd.DataFrame(sessions.user_id.value_counts(sort = False))
# Set action_count equal to user_id
sessions_summary['action_count'] = sessions_summary.user_id
# Set user_id equal to the index
sessions_summary['user_id'] = sessions_summary.index
# Rest the index
sessions_summary = sessions_summary.reset_index(drop = True)


# Looks good, now let's add some features!

# In[ ]:

sessions_summary.head()


# In[ ]:

# user_duration is the sums of secs_elapsed for each user
user_duration = pd.DataFrame(sessions.groupby('user_id').secs_elapsed.sum())
user_duration['user_id'] = user_duration.index
# Merge dataframes
sessions_summary = sessions_summary.merge(user_duration)
# Create new feature, 'duration', to equal secs_elapsed
sessions_summary['duration'] = sessions_summary.secs_elapsed
sessions_summary = sessions_summary.drop("secs_elapsed", axis = 1)


# In[ ]:

sessions_summary.head()


# In[ ]:

# This function finds the most common value, for a specific feature, for each user.
def most_frequent_value(merge_df, feature):
    # Group by the users and find the value counts of the feature
    new_df = pd.DataFrame(sessions.groupby('user_id')[feature].value_counts())
    # The index is a tuple, and we need to seperate it, so let's create a new feature from it.
    new_df['index_tuple'] = new_df.index
    # The new columns are the features created from the tuple.
    new_columns = ['user_id',feature]
    for n,col in enumerate(new_columns):
        new_df[col] = new_df.index_tuple.apply(lambda index_tuple: index_tuple[n])
    
    # Drop the old index (the tuple index)
    new_df = new_df.reset_index(drop = True)
    # Drop the unneeded feature
    new_df = new_df.drop('index_tuple', axis = 1)
    # Select the first value for each user, its most common
    new_df = new_df.groupby('user_id').first()
    
    # Set user_id equal to the index, then reset the index
    new_df['user_id'] = new_df.index
    new_df = new_df.reset_index(drop = True)
    
    merge_df = merge_df.merge(new_df)
    
    return merge_df


# In[ ]:

# For each categorical feature in sessions, find the most common value for each user.
sessions_feature = ['action','action_type','action_detail','device_type']

for feature in sessions_feature:
    sessions_summary = most_frequent_value(sessions_summary, feature)
    print("{} is complete.".format(feature))


# In[ ]:

sessions_summary.head()


# In[ ]:

# This function finds the number of unique values of a feature for each user.
def unique_features(feature, feature_name, merge_df):
    # Create a dataframe by grouping the users and the feature
    unique_feature = pd.DataFrame(sessions.groupby('user_id')[feature].unique())
    unique_feature['user_id'] = unique_feature.index
    unique_feature = unique_feature.reset_index(drop = True)
    # Create a new feature equal to the number of unique features for each user
    unique_feature[feature_name] = unique_feature[feature].map(lambda x: len(x))
    # Drop the needed feature
    unique_feature = unique_feature.drop(feature, axis = 1)
    merge_df = merge_df.merge(unique_feature, on = 'user_id')
    return merge_df


# In[ ]:

# Apply unique_features to each of the categorical features in sessions
sessions_summary = unique_features('action', 'unique_actions', sessions_summary)
print("action is complete.")
sessions_summary = unique_features('action_type', 'unique_action_types', sessions_summary)
print("action_type is complete.")
sessions_summary = unique_features('action_detail', 'unique_action_details', sessions_summary)
print("action_detail is complete.")
sessions_summary = unique_features('device_type', 'unique_device_types', sessions_summary)
print("device_type is complete.")


# In[ ]:

sessions_summary.head()


# In[ ]:

# Find the maximum and minimum secs_elapsed/duration for each user in sessions.
max_durations = pd.DataFrame(sessions.groupby(['user_id'], as_index = False)['secs_elapsed'].max())
sessions_summary = sessions_summary.merge(max_durations, on = 'user_id')
sessions_summary['max_duration'] = sessions_summary.secs_elapsed
sessions_summary = sessions_summary.drop('secs_elapsed', axis = 1)

print("max_durations is complete.")

min_durations = pd.DataFrame(sessions.groupby(['user_id'], as_index = False)['secs_elapsed'].min())
sessions_summary = sessions_summary.merge(min_durations, on = 'user_id')
sessions_summary['min_duration'] = sessions_summary.secs_elapsed
sessions_summary = sessions_summary.drop('secs_elapsed', axis = 1)

print("min_durations is complete.")


# In[ ]:

# Find the average duration for each user
sessions_summary['avg_duration'] = sessions_summary.duration / sessions_summary.action_count


# In[ ]:

sessions_summary.head(5)


# In[ ]:

# Add new features based on the type of device that the user used most frequently.
apple_device = ['Mac Desktop','iPhone','iPdad Tablet','iPodtouch']
desktop_device = ['Mac Desktop','Windows Desktop','Chromebook','Linux Desktop']
tablet_device = ['Android App Unknown Phone/Tablet','iPad Tablet','Tablet']
mobile_device = ['Android Phone','iPhone','Windows Phone','Blackberry','Opera Phone']

device_types = {'apple_device': apple_device, 
                'desktop_device': desktop_device,
                'tablet_device': tablet_device,
                'mobile_device': mobile_device}

for device in device_types:
    sessions_summary[device] = 0
    sessions_summary.loc[sessions_summary.device_type.isin(device_types[device]), device] = 1


# In[ ]:

sessions_summary.head()


# In[ ]:

# Check if there are any null values before merging with train.
sessions_summary.isnull().sum()


# In[ ]:

print(sessions_summary.shape)
print(train.shape)
print(test.shape)


# In[ ]:

# Merge train and test with sessions_summary
train1 = train.merge(sessions_summary, left_on = train['id'], right_on = sessions_summary['user_id'], how = 'inner')
# train2 is equal to the users that are not in train1
train2 = train[~train.id.isin(train1.id)]
train = pd.concat([train1, train2])

test1 = test.merge(sessions_summary, left_on = test['id'], right_on = sessions_summary['user_id'], how = 'inner')
# test2 is equal to the users that are not in test1
test2 = test[~test.id.isin(test1.id)]
test = pd.concat([test1, test2])


# The next step is to transform the features so that they are ready for training the neural network.

# ## Feature Engineering

# In[ ]:

# Concatenate train and test because all transformations need to happen to both dataframes.
df = pd.concat([train,test])


# In[ ]:

df.head()


# In[ ]:

df.shape


# In[ ]:

df.isnull().sum()


# In[ ]:

# We don't need this because we already have id and it has 0 null values.
df = df.drop('user_id', axis = 1)


# Since there are many users that do not appear in the sessions dataframe, all of their values are NaN. Let's sort out those nulls values first.

# In[ ]:

def missing_session_data_cat(feature):
    return df[feature].fillna("missing")

def missing_session_data_cont(feature):
    return df[feature].fillna(0)


# In[ ]:

session_features_cat = ['action','action_detail','action_type','device_type']
session_features_cont = ['action_count','apple_device','desktop_device','mobile_device','tablet_device',
                         'duration','avg_duration','max_duration','min_duration','unique_action_details',
                         'unique_action_types','unique_actions','unique_device_types']

for feature in session_features_cat:
    df[feature] = missing_session_data_cat(feature)
    
for feature in session_features_cont:
    df[feature] = missing_session_data_cont(feature)


# That removed most of the null values!

# In[ ]:

df.isnull().sum()


# In[ ]:

df.action_count.describe()


# In[ ]:

df[df.action_count > 0].action_count.describe()


# In[ ]:

plt.hist(df[df.action_count > 0].action_count)
plt.yscale('log')
plt.show()


# In[ ]:

# Group action_count into quartiles.
df['action_count_quartile'] = df.action_count.map(lambda x: 0 if x == 0 else (
                                                            1 if x <= 17 else (
                                                            2 if x <= 43 else (
                                                            3 if x <= 97 else 4))))


# In[ ]:

df[df.age.notnull()].age.describe()


# In[ ]:

plt.hist(df[df.age <= 100].age, bins = 80)
plt.show()


# No one is 2014 years old. If anyone is older than 80, let's bring their age down to 80...I'm sure some of them wouldn't mind that.

# In[ ]:

df.loc[df.age > 80, 'age'] = 80


# In[ ]:

df[df.age.notnull()].age.describe()


# Let's see if there is a feature that is correlated with age, to help find good values for the nulls

# In[ ]:

for feature in df.columns:
    if(df[feature].dtype == float or df[feature].dtype == int):
        correlation = stats.pearsonr(df[df.age.notnull()].age, df[df.age.notnull()][feature])
        print("Correlation with {} = {}".format(feature, correlation)) 


# Unfortunately, nothing is really correlated with age. Since there are too many missing values for age, I'm going to set the missing values equal to the median, 33.

# In[ ]:

# Create age_group before filling in the nulls, so that the distribution is not altered.
df['age_group'] = df.age.map(lambda x: 0 if math.isnan(x) else (
                                       1 if x < 18 else (
                                       2 if x <= 33 else (
                                       3 if x <= 42 else 4))))


# In[ ]:

df.age = df.age.fillna(33)


# In[ ]:

df.age.isnull().sum()


# In[ ]:

plt.figure(figsize=(8,4))
plt.hist(df.duration, bins = 100)
plt.title("Duration")
plt.xlabel("Duration")
plt.ylabel("Number of Users")
plt.yscale('log')
plt.show()


# In[ ]:

plt.figure(figsize=(8,4))
plt.hist(df.avg_duration, bins = 100)
plt.title("Average Duration")
plt.xlabel("Average Duration")
plt.ylabel("Number of Users")
plt.yscale('log')
plt.show()


# In[ ]:

print(df.duration.describe())
print()
print(df.avg_duration.describe())


# In[ ]:

print(np.percentile(df.duration, 50))
print(np.percentile(df.duration, 51))
print(np.percentile(df.duration, 75))
print()
print(np.percentile(df.avg_duration, 50))
print(np.percentile(df.avg_duration, 51))
print(np.percentile(df.avg_duration, 75))


# In[ ]:

# Divide users into 3 equal-ish groups
df['duration_group'] = df.duration.map(lambda x: 0 if x == 0 else (
                                                 1 if x <= 877166.25 else 2))

df['avg_duration_group'] = df.avg_duration.map(lambda x: 0 if x == 0 else (
                                                         1 if x <= 16553.7889474 else 2))


# In[ ]:

print(df.duration_group.value_counts())
print()
print(df.avg_duration_group.value_counts())


# There are too many unknowns to try to group them with a gender. I'm going to leave OTHER because it could/probably represent(s) people who do not identify as either male/female.

# In[ ]:

df.gender.value_counts()


# In[ ]:

df.mobile_device.value_counts()


# In[ ]:

df.signup_flow.value_counts()


# In[ ]:

# If signup_flow == 0, signup_flow_simple == 0
# If signup_flow > 0, signup_flow_simple == 1
df['signup_flow_simple'] = df.signup_flow.map(lambda x: 0 if x == 0 else 1)


# In[ ]:

df['signup_flow_simple'].value_counts()


# In[ ]:

df.tablet_device.value_counts()


# In[ ]:

# Convert dates to datetime for manipulation
df.date_account_created = pd.to_datetime(df.date_account_created, format='%Y-%m-%d')
df.date_first_booking = pd.to_datetime(df.date_first_booking, format='%Y-%m-%d')


# In[ ]:

# Check to make sure the date range makes sense.
print(df.date_account_created.min())
print(df.date_account_created.max())
print()
print(df.date_first_booking.min())
print(df.date_first_booking.max())


# In[ ]:

# calendar contains more years of information than we need.
calendar = USFederalHolidayCalendar()
# Set holidays equal to the holidays in our date range.
holidays = calendar.holidays(start = df.date_account_created.min(), 
                             end = df.date_first_booking.max())

# us_bd contains more years of information than we need.
us_bd = CustomBusinessDay(calendar = USFederalHolidayCalendar())
# Set business_days equal to the work days in our date range.
business_days = pd.DatetimeIndex(start = df.date_account_created.min(), 
                                 end = df.date_first_booking.max(), 
                                 freq = us_bd)


# In[ ]:

# Create date features
df['year_account_created'] = df.date_account_created.dt.year
df['month_account_created'] = df.date_account_created.dt.month
df['weekday_account_created'] = df.date_account_created.dt.weekday
df['business_day_account_created'] = df.date_account_created.isin(business_days)
df['business_day_account_created'] = df.business_day_account_created.map(lambda x: 1 if x == True else 0)
df['holiday_account_created'] = df.date_account_created.isin(holidays)
df['holiday_account_created'] = df.holiday_account_created.map(lambda x: 1 if x == True else 0)

df['year_first_booking'] = df.date_first_booking.dt.year
df['month_first_booking'] = df.date_first_booking.dt.month
df['weekday_first_booking'] = df.date_first_booking.dt.weekday
df['business_day_first_booking'] = df.date_first_booking.isin(business_days)
df['business_day_first_booking'] = df.business_day_first_booking.map(lambda x: 1 if x == True else 0)
df['holiday_first_booking'] = df.date_first_booking.isin(holidays)
df['holiday_first_booking'] = df.holiday_first_booking.map(lambda x: 1 if x == True else 0)

# Drop unneeded features
df = df.drop(["date_first_booking","date_account_created"], axis = 1)


# In[ ]:

df.isnull().sum()


# In[ ]:

# Set nulls values equal to one less than the minimum.
# I could set the nulls to 0, but the scale would be ugly when we normalize the features.
df.year_first_booking = df.year_first_booking.fillna(min(df.year_first_booking) - 1)
df.month_first_booking = df.month_first_booking.fillna(min(df.month_first_booking) - 1)
df.weekday_first_booking += 1
df.weekday_first_booking = df.weekday_first_booking.fillna(0)


# In[ ]:

df.isnull().sum()


# In[ ]:

df.first_affiliate_tracked.value_counts()


# For the missing values for "first_affiliate_tracked" I am going to set these equal to "untracked". Not only is this the most common value, but it makes sense that if we are missing data on these people that they would not have been tracked.

# In[ ]:

df.first_affiliate_tracked = df.first_affiliate_tracked.fillna("untracked")


# In[ ]:

df.isnull().sum()


# Everything is all clean (the null values in 'country_destination' belong to the testing data). Now let's explore the categorical features that might have too many values and reduce that number before we do one-hot encoding.

# In[ ]:

df.head()


# In[ ]:

df.first_browser.value_counts()


# In[ ]:

# Create a new feature for those using mobile browsers
mobile_browsers = ['Mobile Safari','Chrome Mobile','IE Mobile','Mobile Firefox','Android Browser']
df.loc[df.first_browser.isin(mobile_browsers), "first_browser"] = "Mobile"


# In[ ]:

# The cut_off is set at 0.5% of the data. If a value is not common enough, it will be grouped into something generic.
cut_off = 1378

other_browsers = []
for browser, count in df.first_browser.value_counts().iteritems():
    if count < cut_off:
        other_browsers.append(browser)
   
df.loc[df.first_browser.isin(other_browsers), "first_browser"] = "Other"

print(other_browsers)


# In[ ]:

df.first_browser.value_counts()


# In[ ]:

df.language.value_counts()


# I think that language might be a more important feature than some others, so I will decrease the cut off to 275, or 0.1% of the data.

# In[ ]:

other_languages = []
for language, count in df.language.value_counts().iteritems():
    if count < 275:
        other_languages.append(language)
    
print(other_languages)

df.loc[df.language.isin(other_languages), "language"] = "Other"


# In[ ]:

df.language.value_counts()


# In[ ]:

# New feature for languages that are not English.
df['not_English'] = df.language.map(lambda x: 0 if x == 'en' else 1)


# In[ ]:

df.action.value_counts()


# In[ ]:

other_actions = []
for action, count in df.action.value_counts().iteritems():
    if count < cut_off:
        other_actions.append(action)
    
print(other_actions)

df.loc[df.action.isin(other_actions), "action"] = "Other"


# In[ ]:

df.action.value_counts()


# In[ ]:

df.action_detail.value_counts()


# In[ ]:

other_action_details = []
for action_detail, count in df.action_detail.value_counts().iteritems():
    if count < cut_off:
        other_action_details.append(action_detail)
    
print(other_action_details)

df.loc[df.action_detail.isin(other_action_details), "action_detail"] = "Other"


# In[ ]:

df.action_detail.value_counts()


# In[ ]:

df.action_type.value_counts()


# In[ ]:

other_action_types = []
for action_type, count in df.action_type.value_counts().iteritems():
    if count < 1378:
        other_action_types.append(action_type)
    
print(other_action_types)

df.loc[df.action_type.isin(other_action_types), "action_type"] = "Other"


# In[ ]:

df.action_type.value_counts()


# In[ ]:

df.affiliate_provider.value_counts()


# In[ ]:

other_affiliate_providers = []
for affiliate_provider, count in df.affiliate_provider.value_counts().iteritems():
    if count < cut_off:
        other_affiliate_providers.append(affiliate_provider)
    
print(other_affiliate_providers)

df.loc[df.affiliate_provider.isin(other_affiliate_providers), "affiliate_provider"] = "other"


# In[ ]:

df.affiliate_provider.value_counts()


# In[ ]:

df.device_type.value_counts()


# In[ ]:

other_device_types = []
for device_type, count in df.device_type.value_counts().iteritems():
    if count < 1378:
        other_device_types.append(device_type)
    
print(other_device_types)

df.loc[df.device_type.isin(other_device_types), "device_type"] = "Other"


# In[ ]:

df.device_type.value_counts()


# In[ ]:

df.signup_method.value_counts()


# In[ ]:

# Create a new dataframe for the labels
labels = pd.DataFrame(df.country_destination)
df = df.drop("country_destination", axis = 1)


# In[ ]:

labels.head()


# In[ ]:

# Drop id since it is no longer needed.
df = df.drop('id', axis = 1)


# In[ ]:

# Group all features as either continuous (cont) or categorical (cat)
cont_features = []
cat_features = []

for feature in df.columns:
    if df[feature].dtype == float or df[feature].dtype == int:
        cont_features.append(feature)
    elif df[feature].dtype == object:
        cat_features.append(feature)


# In[ ]:

# Check to ensure that we have all of the features
print(cat_features)
print()
print(cont_features)
print()
print(len(cat_features) + len(cont_features))
print(df.shape[1])


# In[ ]:

# Although dates have continuous values, they should be treated as categorical features.
date_features = ['year_account_created','month_account_created','weekday_account_created',
                      'year_first_booking','month_first_booking','weekday_first_booking']
for feature in date_features:
    cont_features.remove(feature)
    cat_features.append(feature)


# In[ ]:

for feature in cat_features:
    # Create dummies of each value of a categorical feature
    dummies = pd.get_dummies(df[feature], prefix = feature, drop_first = False)
    # Drop the unneeded feature
    df = df.drop(feature, axis = 1)
    df = pd.concat([df, dummies], axis=1)
    print("{} is complete".format(feature))


# In[ ]:

min_max_scaler = preprocessing.MinMaxScaler()
# Normalize the continuous features
for feature in cont_features:
    df.loc[:,feature] = min_max_scaler.fit_transform(df[feature])


# In[ ]:

df.head()


# In[ ]:

# Split df into training and testing data
df_train = df[:len(train)]
df_test = df[len(train):]

# Shorten labels to length of the training data
y = labels[:len(train)]


# In[ ]:

# Create dummy features for each country
y_dummies = pd.get_dummies(y, drop_first = False)
y = pd.concat([y, y_dummies], axis=1)
y = y.drop("country_destination", axis = 1)
y.head()


# In[ ]:

print(df_train.shape)
print(df_test.shape)
print(y.shape)


# In[ ]:

# Take a look to see how common each country is.
train.country_destination.value_counts() 


# In[ ]:

# Find the order of the features
y.columns


# Due to the imbalance in the data, we are going to set the sum of each feature equal to each other. This will help the neural network to train because it won't be biased to reducing the NDF errors since that would have the greatest effect in the cost function.

# In[ ]:

y[y.columns[0]] *= len(y)/539
y[y.columns[1]] *= len(y)/1428
y[y.columns[2]] *= len(y)/1061
y[y.columns[3]] *= len(y)/2249
y[y.columns[4]] *= len(y)/5023
y[y.columns[5]] *= len(y)/2324
y[y.columns[6]] *= len(y)/2835
y[y.columns[7]] *= len(y)/124543
y[y.columns[8]] *= len(y)/762
y[y.columns[9]] *= len(y)/217
y[y.columns[10]] *= len(y)/62376
y[y.columns[11]] *= len(y)/10094


# In[ ]:

# Check the sum of each feature
totals = []
for i in range(12):
    totals.append(sum(y[y.columns[i]]))
totals


# In[ ]:

x_train, x_test, y_train, y_test = train_test_split(df_train, y, test_size = 0.2, random_state = 2)


# In[ ]:

# Tensorflow needs the data in matrices
inputX = x_train.as_matrix()
inputY = y_train.as_matrix()
inputX_test = x_test.as_matrix()
inputY_test = y_test.as_matrix()


# In[ ]:

# Number of input nodes/number of features.
input_nodes = 186

# Multiplier maintains a fixed ratio of nodes between each layer.
mulitplier = 1.33

# Number of nodes in each hidden layer
hidden_nodes1 = 50
hidden_nodes2 = round(hidden_nodes1 * mulitplier)

# Percent of nodes to keep during dropout.
pkeep = tf.placeholder(tf.float32)


# In[ ]:

# The standard deviation when setting the values for the weights.
std = 1

#input
features = tf.placeholder(tf.float32, [None, input_nodes])

#layer 1
W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev = std))
b1 = tf.Variable(tf.zeros([hidden_nodes1]))
y1 = tf.nn.sigmoid(tf.matmul(features, W1) + b1)

#layer 2
W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev = std))
b2 = tf.Variable(tf.zeros([hidden_nodes2]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)
#y2 = tf.nn.dropout(y2, pkeep)

#layer 3
W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, 12], stddev = std)) 
b3 = tf.Variable(tf.zeros([12]))
y3 = tf.nn.softmax(tf.matmul(y2, W3) + b3)

#output
predictions = y3
labels = tf.placeholder(tf.float32, [None, 12])


# In[ ]:

#Parameters
training_epochs = 3000
training_dropout = 0.6 # Not using dropout led to the best results.
display_step = 10
n_samples = inputY.shape[1]
batch = tf.Variable(0)

learning_rate = tf.train.exponential_decay(
  0.05,              #Base learning rate.
  batch,             #Current index into the dataset.
  len(inputX),       #Decay step.
  0.95,              #Decay rate.
  staircase=False)


# Based on the evaluation method of the Kaggle competition, we are going to check the accuracy of the top prediction and the top 5 predictions for each user.

# In[ ]:

# Determine if the predictions are correct
correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1))
correct_top5 = tf.nn.in_top_k(predictions, tf.argmax(labels, 1), k = 5)

# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_top5 = tf.reduce_mean(tf.cast(correct_top5, tf.float32))

print('Accuracy function created.')

# Cross entropy
cross_entropy = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(predictions,1e-10,1.0)))

# Training loss
loss = tf.reduce_mean(cross_entropy)

#We will optimize our model via AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[ ]:

#Initialize variables and tensorflow session
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)


# In[ ]:

accuracy_summary = [] #Record accuracy values for plot
accuracy_top5_summary = [] #Record accuracy values for plot
loss_summary = [] #Record cost values for plot

test_accuracy_summary = [] #Record accuracy values for plot
test_accuracy_top5_summary = [] #Record accuracy values for plot
test_loss_summary = [] #Record cost values for plot

init = tf.initialize_all_variables()

for i in range(training_epochs):  
    session.run([optimizer], 
                feed_dict={features: inputX, 
                           labels: inputY,
                           pkeep: training_dropout})

    # Display logs per epoch step
    if (i) % display_step == 0:
        train_accuracy, train_accuracy_top5, newLoss = session.run([accuracy,accuracy_top5,loss], 
                                                                   feed_dict={features: inputX, 
                                                                              labels: inputY,
                                                                              pkeep: training_dropout})
        print ("Epoch:", i,
               "Accuracy =", "{:.6f}".format(train_accuracy), 
               "Top 5 Accuracy =", "{:.6f}".format(train_accuracy_top5),
               "Loss = ", "{:.6f}".format(newLoss))
        accuracy_summary.append(train_accuracy)
        accuracy_top5_summary.append(train_accuracy_top5)
        loss_summary.append(newLoss)
        
        test_accuracy,test_accuracy_top5,test_newLoss = session.run([accuracy,accuracy_top5,loss], 
                                                              feed_dict={features: inputX_test, 
                                                                         labels: inputY_test,
                                                                         pkeep: 1})
        print ("Epoch:", i,
               "Test-Accuracy =", "{:.6f}".format(test_accuracy), 
               "Test-Top 5 Accuracy =", "{:.6f}".format(test_accuracy_top5),
               "Test-Loss = ", "{:.6f}".format(test_newLoss))
        test_accuracy_summary.append(test_accuracy)
        test_accuracy_top5_summary.append(test_accuracy_top5)
        test_loss_summary.append(test_newLoss)

print()
print ("Optimization Finished!")
training_accuracy, training_top5_accuracy = session.run([accuracy,accuracy_top5], 
                                feed_dict={features: inputX, labels: inputY, pkeep: training_dropout})
print ("Training Accuracy=", training_accuracy)
print ("Training Top 5 Accuracy=", training_top5_accuracy)
print()
testing_accuracy, testing_top5_accuracy = session.run([accuracy,accuracy_top5], 
                                                       feed_dict={features: inputX_test, 
                                                                  labels: inputY_test,
                                                                  pkeep: 1})
print ("Testing Accuracy=", testing_accuracy)
print ("Testing Top 5 Accuracy=", testing_top5_accuracy)


# In[ ]:

testing_predictions, testing_labels = session.run([tf.argmax(predictions,1), tf.argmax(labels,1)], 
                                                  feed_dict={features: inputX_test,
                                                             labels: inputY_test,
                                                             pkeep: 1})

print(classification_report(testing_labels, testing_predictions, target_names=y.columns))


# In[ ]:

#Plot accuracies and cost summary
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,8))

ax1.plot(accuracy_summary)
ax1.plot(test_accuracy_summary)
ax1.set_title('Top 1 Accuracy')

ax2.plot(accuracy_top5_summary)
ax2.plot(test_accuracy_top5_summary)
ax2.set_title('Top 5 Accuracy')

ax3.plot(loss_summary)
ax3.plot(test_loss_summary)
ax3.set_title('Loss')

plt.xlabel('Epochs (x10)')
plt.show()


# In[ ]:

# Find the probabilities for each prediction
test_final = df_test.as_matrix()
final_probabilities = session.run(predictions, feed_dict={features: test_final,
                                                          pkeep: 1})


# In[ ]:

# Explore some of the predictions
final_probabilities[0]


# In[ ]:

# Encode the labels for the countries
le = LabelEncoder()
fit_labels = le.fit_transform(train.country_destination) 

# Get the ids for the test data
test_getIDs = pd.read_csv("test_users.csv")
testIDs = test_getIDs['id']

ids = []  #list of ids
countries = []  #list of countries
for i in range(len(testIDs)):
    # Select the 5 countries with highest probabilities
    idx = testIDs[i]
    ids += [idx] * 5
    countries += le.inverse_transform(np.argsort(final_probabilities[i])[::-1])[:5].tolist()
    if i % 10000 == 0:
        print ("Percent complete: {}%".format(round(i / len(test),4)*100))

#Generate submission
submission = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'country'])
submission.to_csv('submission.csv',index=False)


# In[ ]:

# Check some of the submissions
submission.head(25)


# In[ ]:

# Compare the submission's distribution to the training data's distribution.
# Given that the data was randomly split, 
# a more equal distribution should lead to better scores in the Kaggle competition.
submission.country.value_counts()


# In[ ]:

train.country_destination.value_counts()


# ## Summary

# Based on Kaggle's evaluation method (https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings#evaluation), the neural network scores just under 0.86 when all of the training data is used. The winning submission scored 0.88697 and the sample submission scored 0.68411. Looking at the leaderboard and some kernels, the most common algorithm for the better scores was XGBoost. Given that the purpose of this analyis was to further my knowledge of TensorFlow (in addition to the other aspects of machine learning - i.e. feature engineering, cleaning data, etc.), I do not feel the need to use XGBoost to try to make a better prediction. I am rather pleased with this model on a whole, given its ability to accurately predict which country a user will make his/her first trip in. The 'lazy' prediction method would be to use the top and top 5 most common countries for the predictions. This would equal an accuracy score of 58.35% for the top predictions and 95.98% for the top 5 predictions. For the testing data, my top predictions scored a higher accuracy of 62.37%, as well as for the top 5 predictions, at 95.99%. My predictions are also more useful since they make use of all twelve countries, instead of just the five most common.

# In[ ]:



