# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppressing warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Load the dataset
zomato_real = pd.read_csv("C:/Users/Lenovo/Downloads/zomato.csv")

# Displaying the first few rows and dataset info
zomato_real.head() 
zomato_real.info()

# Deleting unnecessary columns
zomato = zomato_real.drop(['url', 'dish_liked', 'phone'], axis=1) 

# Removing duplicates
zomato.drop_duplicates(inplace=True)

# Removing NaN values
zomato.dropna(how='any', inplace=True)

# Changing column names
zomato = zomato.rename(columns={'approx_cost(for two people)': 'cost', 
                                'listed_in(type)': 'type', 
                                'listed_in(city)': 'city'})

# Data transformations for 'cost' column
# Removing commas and converting the 'cost' column to float
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',', '')).astype(float)

# Scaling 'cost' between 1 and 10
scaler = MinMaxScaler(feature_range=(1, 10))
zomato['cost'] = scaler.fit_transform(zomato[['cost']])

# Removing 'NEW' and '-' from 'rate' column
zomato = zomato.loc[zomato.rate != 'NEW']
zomato = zomato.loc[zomato.rate != '-'].reset_index(drop=True)
zomato.rate = zomato.rate.apply(lambda x: x.replace('/5', '') if isinstance(x, str) else x).str.strip().astype('float')

# Title case for restaurant names
zomato.name = zomato.name.apply(lambda x: x.title())

# Replacing 'Yes'/'No' with True/False for 'online_order' and 'book_table'
zomato.online_order.replace(('Yes', 'No'), (True, False), inplace=True)
zomato.book_table.replace(('Yes', 'No'), (True, False), inplace=True)

# Adding Mean Rating for each restaurant
restaurants = zomato['name'].unique()
zomato['Mean Rating'] = 0

for restaurant in restaurants:
    zomato['Mean Rating'][zomato['name'] == restaurant] = zomato['rate'][zomato['name'] == restaurant].mean()

# Scaling 'Mean Rating' between 1 and 5
scaler = MinMaxScaler(feature_range=(1, 5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

# Saving the cleaned dataset to a CSV file
zomato.to_csv('Cleaned_data.csv', index=False)

# Displaying a sample of the cleaned data
zomato.sample(3)
