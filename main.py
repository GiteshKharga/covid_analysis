import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = 'https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv'
df = pd.read_csv(url)

# Display data types
print(df.dtypes)

# Display first few rows
print(df.head())  

# Display shape of the DataFrame
print("Shape:", df.shape)

# Summary statistics
print(df.describe())
print(df.dtypes.value_counts)
print (" unique location:", df['location'].nunique())
print("continent value counts:\n", df['continent'].value_counts())
print("total cases:",df['total_cases'].max())
print("mean total cases:", df['total_cases'].mean())
print("quartiles of total deaths:",df['total_deaths'].quantile([0.25,0.5,0.75]))
print("max hdi row:\n",df.loc[df['human_development_index'].idxmax()])
print("min gdp row:\n",df.loc[df['gdp_per_capita'].idxmin()])

columns_to_keep = ['continent','location', 'date', 'total_cases', 'total_deaths', 'human_development_index', 'gdp_per_capita']
df= df[columns_to_keep]
df=df.drop_duplicates()
print(df.isnull().sum())
df=df.dropna(subset=['continent'])
df=df.fillna(0)
