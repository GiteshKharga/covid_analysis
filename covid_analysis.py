import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# URL of the dataset - Updated to a working URL
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

# Import the dataset using pandas
print("Loading dataset...")
df = pd.read_csv(url)

# Filter the dataframe with specific columns
selected_columns = [
    'continent',
    'location',
    'date',
    'total_cases',
    'total_deaths',
    'gdp_per_capita',
    'human_development_index'
]

# Update the dataframe with selected columns
print("Filtering columns...")
df = df[selected_columns]

# Data Cleaning Steps
print("\n=== Data Cleaning Steps ===")

# a. Remove all duplicate observations
print("\na. Removing duplicate observations...")
initial_rows = len(df)
df = df.drop_duplicates()
rows_after_dedup = len(df)
print(f"Removed {initial_rows - rows_after_dedup} duplicate rows")

# b. Find missing values in all columns
print("\nb. Missing values in each column:")
missing_values = df.isnull().sum()
print(missing_values)

# Calculate percentage of missing values
missing_percentage = (missing_values / len(df)) * 100
print("\nPercentage of missing values in each column:")
print(missing_percentage)

# c. Remove all observations where continent column value is missing
print("\nc. Removing rows with missing continent values...")
initial_rows = len(df)
df = df.dropna(subset=['continent'])
rows_after_cleaning = len(df)
print(f"Removed {initial_rows - rows_after_cleaning} rows with missing continent values")

# Display the cleaned dataframe information
print("\nCleaned DataFrame Information:")
print(df.info())

# Display first few rows of the cleaned dataframe
print("\nFirst 5 rows of the cleaned dataset:")
print(df.head())

# Display the shape of the cleaned dataframe
print(f"\nShape of cleaned dataframe: {df.shape}")

# a. Find number of rows & columns
print("\nDataset Dimensions:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# b. Data types of columns
print("\nData Types of Columns:")
print(df.dtypes)

# c. Info & describe of data in dataframe
print("\nDetailed Dataset Information:")
print(df.info())

print("\nStatistical Summary of the Dataset:")
print(df.describe())

# a. Find count of unique values in location column
print("\na. Count of unique values in location column:")
print(f"Number of unique locations: {df['location'].nunique()}")
print("\nUnique locations:")
print(df['location'].unique())

# b. Find which continent has maximum frequency
print("\nb. Continent with maximum frequency:")
continent_counts = df['continent'].value_counts()
print(continent_counts)
print(f"\nContinent with maximum frequency: {continent_counts.index[0]} with {continent_counts.iloc[0]} occurrences")

# c. Find maximum & mean value in 'total_cases'
print("\nc. Statistics for total_cases:")
print(f"Maximum total cases: {df['total_cases'].max():,.0f}")
print(f"Mean total cases: {df['total_cases'].mean():,.2f}")

# d. Find quartile values in 'total_deaths'
print("\nd. Quartile values for total_deaths:")
quartiles = df['total_deaths'].quantile([0.25, 0.50, 0.75])
print(f"25% quartile: {quartiles[0.25]:,.0f}")
print(f"50% quartile (median): {quartiles[0.50]:,.0f}")
print(f"75% quartile: {quartiles[0.75]:,.0f}")

# e. Find which continent has maximum 'human_development_index'
print("\ne. Continent with maximum human_development_index:")
max_hdi_continent = df.loc[df['human_development_index'].idxmax()]
print(f"Continent: {max_hdi_continent['continent']}")
print(f"Human Development Index: {max_hdi_continent['human_development_index']:.3f}")

# f. Find which continent has minimum 'gdp_per_capita'
print("\nf. Continent with minimum gdp_per_capita:")
min_gdp_continent = df.loc[df['gdp_per_capita'].idxmin()]
print(f"Continent: {min_gdp_continent['continent']}")
print(f"GDP per capita: ${min_gdp_continent['gdp_per_capita']:,.2f}")

# === Date Time Format ===
print("\n=== Date Time Format ===")
# a. Convert date column in datetime format
print("Converting 'date' column to datetime format...")
df['date'] = pd.to_datetime(df['date'])

# b. Create new column 'month' after extracting month data from date column
df['month'] = df['date'].dt.month
print("Added 'month' column extracted from 'date'.")

# Display first few rows to verify changes
print("\nFirst 5 rows after date conversion and month extraction:")
print(df.head())

# === Data Aggregation ===
print("\n=== Data Aggregation ===")
# a. Find max value in all columns using groupby on 'continent'
df_groupby = df.groupby('continent').max().reset_index()
print("Aggregated max values by continent:")
print(df_groupby)

# === Feature Engineering ===
print("\n=== Feature Engineering ===")
# a. Create new feature 'total_deaths_to_total_cases'
df_groupby['total_deaths_to_total_cases'] = df_groupby['total_deaths'] / df_groupby['total_cases']
print("\nAdded new feature 'total_deaths_to_total_cases'")
print(df_groupby[['continent', 'total_deaths_to_total_cases']])

# === Data Visualization ===
print("\n=== Data Visualization ===")
# Set the style for all plots
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This will set seaborn's default theme

# a. Univariate analysis on 'gdp_per_capita'
plt.figure(figsize=(10, 6))
sns.histplot(data=df_groupby, x='gdp_per_capita', bins=10)
plt.title('Distribution of GDP per Capita by Continent')
plt.xlabel('GDP per Capita')
plt.ylabel('Count')
plt.savefig('gdp_distribution.png')
plt.close()

# b. Scatter plot of 'total_cases' & 'gdp_per_capita'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_groupby, x='gdp_per_capita', y='total_cases', hue='continent')
plt.title('Total Cases vs GDP per Capita by Continent')
plt.xlabel('GDP per Capita')
plt.ylabel('Total Cases')
plt.savefig('cases_vs_gdp.png')
plt.close()

# c. Pairplot on df_groupby dataset
plt.figure(figsize=(12, 8))
sns.pairplot(df_groupby, hue='continent')
plt.savefig('pairplot.png')
plt.close()

# d. Bar plot of 'continent' with 'total_cases'
plt.figure(figsize=(12, 6))
sns.catplot(data=df_groupby, x='continent', y='total_cases', kind='bar')
plt.title('Total Cases by Continent')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cases_by_continent.png')
plt.close()

# Save the df_groupby dataframe
print("\n=== Saving Data ===")
df_groupby.to_csv('covid_analysis_results.csv', index=False)
print("Saved df_groupby to 'covid_analysis_results.csv'")

# Display final dataframe info
print("\nFinal DataFrame Information:")
print(df_groupby.info()) 