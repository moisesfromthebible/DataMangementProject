import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load and Clean Election Data
election_data = pd.read_csv("countypres_2000-2020.csv")
election_data_cleaned = election_data[
    ['year', 'state', 'state_po', 'county_name', 'county_fips', 'candidate', 'party', 'candidatevotes', 'totalvotes']
].dropna().rename(columns={
    'state_po': 'state_abbreviation',
    'county_fips': 'fips_code',
    'candidatevotes': 'votes_candidate',
    'totalvotes': 'votes_total'
})
print("Election Data Cleaned (first 5 rows):")
print(election_data_cleaned.head())

# Load and Clean Unemployment Data
unemployment_data = pd.read_csv("Unemployment(UnemploymentMedianIncome).csv")
unemployment_data_cleaned = unemployment_data.dropna()
if 'State_FIPS' in unemployment_data_cleaned.columns and 'County_FIPS' in unemployment_data_cleaned.columns:
    unemployment_data_cleaned['fips_code'] = (
            unemployment_data_cleaned['State_FIPS'].astype(str).str.zfill(2) +
            unemployment_data_cleaned['County_FIPS'].astype(str).str.zfill(3)
    ).astype(int)
    unemployment_data_cleaned = unemployment_data_cleaned.drop(columns=['State_FIPS', 'County_FIPS'], errors='ignore')

columns_to_remove = [
    'Rural_Urban_Continuum_Code_2013',
    'Urban_Influence_Code_2013',
    'Metro_2013',
    'Median_Household_Income_2021',
    'Med_HH_Income_Percent_of_State_Total_2021'
]
unemployment_data_cleaned = unemployment_data_cleaned.drop(columns=columns_to_remove, errors='ignore')
print("Unemployment Data Cleaned (first 5 rows):")
print(unemployment_data_cleaned.head())

# Load and Clean Population Data
pop_data = pd.read_csv('combined_population_data.csv', encoding='utf-8')
pop_data['fips_code'] = (
        pop_data['STATE'].astype(str).str.zfill(2) + pop_data['COUNTY'].astype(str).str.zfill(3)
).astype(int)
pop_data = pop_data.drop(columns=['STATE', 'COUNTY', 'STNAME'], errors='ignore')
print("Population Data after adding fips_code:")
print(pop_data.head())

# Load and Clean Education Data
education_df = pd.read_excel("Education.xlsx", engine='openpyxl', header=3)
education_df = education_df.rename(columns={
    'FIPS Code': 'fips_code',
    'State': 'state',
    'Area name': 'county_name'
})
education_df['fips_code'] = pd.to_numeric(education_df['fips_code'], errors='coerce')
education_df['county_name'] = education_df['county_name'].str.replace(' County', '', regex=False)
print("Education Data Cleaned (first 5 rows):")
print(education_df.head())

# Insert Data into SQLite Database
db_path = "election_data.sqlite"
conn = sqlite3.connect(db_path)

election_data_cleaned.to_sql("election_data", conn, if_exists="replace", index=False)
unemployment_data_cleaned.columns = [c.replace(' ', '_') for c in unemployment_data_cleaned.columns]
unemployment_data_cleaned.to_sql("unemployment_data", conn, if_exists="replace", index=False)
pop_data.to_sql("population_data", conn, if_exists="replace", index=False)
education_df.to_sql("education_data", conn, if_exists="replace", index=False)

conn.commit()
conn.close()
print("All datasets inserted into the database successfully.")

# Query Data and Prepare for Modeling
conn = sqlite3.connect(db_path)
query = """
SELECT 
    e.fips_code,
    e.year,
    e.state,
    e.county_name,
    SUM(CASE WHEN e.party='DEMOCRAT' THEN e.votes_candidate ELSE 0 END) as dem_votes,
    SUM(CASE WHEN e.party='REPUBLICAN' THEN e.votes_candidate ELSE 0 END) as rep_votes,
    p.*,  -- Population data
    u.*,  -- Unemployment data
    ed.*  -- Education data
FROM election_data e
LEFT JOIN population_data p ON e.fips_code = p.fips_code
LEFT JOIN unemployment_data u ON e.fips_code = u.fips_code
LEFT JOIN education_data ed ON e.fips_code = ed.fips_code
WHERE e.year=2020
GROUP BY e.fips_code, e.year, e.state, e.county_name
"""
model_df = pd.read_sql_query(query, conn)
conn.close()
print("Model DataFrame created successfully:")
print(model_df.head())

# Consolidate POPESTIMATE features into one feature (average)
pop_estimate_columns = [col for col in model_df.columns if 'POPESTIMATE' in col]
model_df['avg_pop_estimate'] = model_df[pop_estimate_columns].mean(axis=1)

# Drop the individual POPESTIMATE columns
model_df = model_df.drop(columns=pop_estimate_columns)

# Consolidate Unemployment Rate features into one feature (average)
unemployment_columns = [col for col in model_df.columns if 'Unemployment_rate' in col]
model_df['avg_unemployment_rate'] = model_df[unemployment_columns].mean(axis=1)

# Drop the individual Unemployment Rate columns
model_df = model_df.drop(columns=unemployment_columns)

# Consolidate "Percent of adults with a high school diploma only" features into one feature (average)
high_school_diploma_columns = [
    'Percent of adults with a high school diploma only, 2000',
    'Percent of adults with a high school diploma only, 2018-22',
    'Percent of adults with a high school diploma only, 1980',
    'Percent of adults with a high school diploma only, 1990',
    'Percent of adults with a high school diploma only, 2008-12',
    'Percent of adults with a high school diploma only, 1970'
]

# Compute the average for these columns
model_df['avg_high_school_diploma_only'] = model_df[high_school_diploma_columns].mean(axis=1)

# Drop the individual "high school diploma only" columns after aggregation
model_df = model_df.drop(columns=high_school_diploma_columns)

# Aggregate Bachelor's degree or higher features
bachelor_columns = [col for col in model_df.columns if "bachelor's degree or higher" in col.lower()]
model_df['avg_bachelor_or_higher'] = model_df[bachelor_columns].mean(axis=1)
model_df = model_df.drop(columns=bachelor_columns)

# Aggregate "Four years of college or higher" features
four_years_college_columns = [col for col in model_df.columns if "four years of college or higher" in col.lower()]
model_df['avg_four_years_college_or_higher'] = model_df[four_years_college_columns].mean(axis=1)
model_df = model_df.drop(columns=four_years_college_columns)

# Aggregate "Percent of adults with less than a high school diploma" features
less_than_high_school_columns = [col for col in model_df.columns if "less than a high school diploma" in col.lower()]
model_df['avg_less_than_high_school_diploma'] = model_df[less_than_high_school_columns].mean(axis=1)
model_df = model_df.drop(columns=less_than_high_school_columns)

# Aggregate "Percent of adults completing some college or associate's degree" features
some_college_or_associate_degree_columns = [
    'Percent of adults completing some college or associate\'s degree, 2018-22',
    'Percent of adults completing some college or associate\'s degree, 1990',
    'Percent of adults completing some college or associate\'s degree, 2008-12',
]

# Create the aggregated feature by averaging the related columns
model_df['avg_some_college_or_associate_degree'] = model_df[some_college_or_associate_degree_columns].mean(axis=1)

# Drop the individual "some college or associate's degree" columns after aggregation
model_df = model_df.drop(columns=some_college_or_associate_degree_columns)

# Combine all years of "Percent of adults completing some college (1-3 years)" by averaging
some_college_columns = [
    'Percent of adults completing some college (1-3 years), 1980',
    'Percent of adults completing some college (1-3 years), 1970',
    'Some college (1-3 years), 1980',  # Including variations in the naming
    'Some college (1-3 years), 1970'
]

# Compute the average for these columns
model_df['avg_some_college_1_3_years'] = model_df[some_college_columns].mean(axis=1)

# Drop the individual columns after aggregation
model_df = model_df.drop(columns=some_college_columns)

high_school_diploma_only_columns = [
    'High school diploma only, 2018-22',
    'High school diploma only, 1970',
    'High school diploma only, 1980',
    'High school diploma only, 1990',
    'High school diploma only, 2008-12',
    'High school diploma only, 2000'
]

# Compute the average for these columns
model_df['avg_high_school_diploma_only'] = model_df[high_school_diploma_only_columns].mean(axis=1)

# Drop the individual "high school diploma only" columns after aggregation
model_df = model_df.drop(columns=high_school_diploma_only_columns)

# Consolidate "Some college or associate's degree" features into one feature (average)
some_college_or_associate_degree_columns = [
    'Percent of adults completing some college or associate\'s degree, 2000',
    'Some college or associate\'s degree, 2018-22',
    'Some college or associate\'s degree, 1990',
    'Some college or associate\'s degree, 2008-12',
    'Some college or associate\'s degree, 2000'
]

# Compute the average for these columns
model_df['avg_some_college_or_associate_degree'] = model_df[some_college_or_associate_degree_columns].mean(axis=1)

# Drop the individual "Some college or associate's degree" columns after aggregation
model_df = model_df.drop(columns=some_college_or_associate_degree_columns)

# Continue with the rest of the data processing...


# Create Target Variable and Feature Set
model_df['dem_share'] = model_df['dem_votes'] / (model_df['dem_votes'] + model_df['rep_votes'])
id_cols = ['fips_code', 'year', 'state', 'county_name', 'dem_votes', 'rep_votes']
X = model_df.drop(columns=id_cols + ['dem_share'], errors='ignore').select_dtypes(include=[np.number]).dropna()
y = model_df.loc[X.index, 'dem_share']

# Remove unwanted features (such as the urban influence codes and fips_code)
columns_to_remove = [
    'FIPS_Code',
    '2003 Urban Influence Code',
    '2013 Urban Influence Code',
    '2023 Rural-urban Continuum Code',
    '2013 Rural-urban Continuum Code'
]

X = X.drop(columns=[col for col in columns_to_remove if col in X.columns], errors='ignore')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Model Performance after removing unwanted features:")
print("MAE:", mae)
print("RMSE:", rmse)

# Feature Importances
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
