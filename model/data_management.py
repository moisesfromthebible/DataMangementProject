import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load Election Data
election_data = pd.read_csv("countypres_2000-2020.csv")

# Clean Election Data
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

unemployment_data = pd.read_csv("Unemployment(UnemploymentMedianIncome).csv")

# Extract column names from the CSV
with open('Unemployment(UnemploymentMedianIncome).csv') as csv_file:
    import csv
    csv_reader = csv.DictReader(csv_file)
    dict_from_csv = dict(list(csv_reader)[0])
    list_of_column_names = list(dict_from_csv.keys())

unemployment_data_cleaned = unemployment_data[list_of_column_names].dropna()

# Create fips_code if possible
if 'State_FIPS' in unemployment_data_cleaned.columns and 'County_FIPS' in unemployment_data_cleaned.columns:
    unemployment_data_cleaned['fips_code'] = unemployment_data_cleaned['State_FIPS'].astype(str).str.zfill(2) + \
                                             unemployment_data_cleaned['County_FIPS'].astype(str).str.zfill(3)
    unemployment_data_cleaned['fips_code'] = unemployment_data_cleaned['fips_code'].astype(int)
    # Drop original FIPS columns if desired
    for c in ['State_FIPS', 'County_FIPS']:
        if c in unemployment_data_cleaned.columns:
            unemployment_data_cleaned = unemployment_data_cleaned.drop(columns=[c])
else:
    print("WARNING: Unemployment data does not have State_FIPS and County_FIPS. Adjust as needed.")

# Remove unneeded columns if they exist
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

# Updated population data cell

pop_data = pd.read_csv('combined_population_data.csv', encoding='utf-8')

# Create fips_code from STATE and COUNTY
pop_data['fips_code'] = pop_data['STATE'].astype(str).str.zfill(2) + pop_data['COUNTY'].astype(str).str.zfill(3)
pop_data['fips_code'] = pop_data['fips_code'].astype(int)

# Drop columns no longer needed
pop_data = pop_data.drop(columns=['STATE','COUNTY','STNAME'], errors='ignore')

print("Population Data after adding fips_code:")
print(pop_data.head())

# Re-insert into DB
conn = sqlite3.connect("election_data.sqlite")
pop_data.to_sql("population_data", conn, if_exists="replace", index=False)
conn.commit()
conn.close()

print("Population data updated and re-inserted with fips_code.")

file_path = "Education.xlsx"

# Headers start at row 4 (0-based indexing: header=3)
education_df = pd.read_excel(file_path, engine='openpyxl', header=3)

# Rename columns
education_df = education_df.rename(columns={
    'FIPS Code': 'fips_code',
    'State': 'state',
    'Area name': 'county_name'
})

# Convert fips_code to numeric
education_df['fips_code'] = pd.to_numeric(education_df['fips_code'], errors='coerce')

# Remove " County" suffix if present
education_df['county_name'] = education_df['county_name'].astype(str).str.replace(' County', '', regex=False)

print("Education Data Cleaned (first 5 rows):")
print(education_df.head())

db_path = "election_data.sqlite"
conn = sqlite3.connect(db_path)

# Election data
election_data_cleaned.to_sql("election_data", conn, if_exists="replace", index=False)

# Unemployment data
unemployment_data_cleaned.columns = [c.replace(' ', '_') for c in unemployment_data_cleaned.columns]
unemployment_data_cleaned.to_sql("unemployment_data", conn, if_exists="replace", index=False)

# Population data
pop_data.to_sql("population_data", conn, if_exists="replace", index=False)

# Education data
education_df.to_sql("education_data", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

print("All datasets inserted into the database successfully.")

conn = sqlite3.connect(db_path)

query = """
SELECT 
    e.fips_code,
    e.year,
    e.state,
    e.county_name,
    SUM(CASE WHEN e.party='DEMOCRAT' THEN e.votes_candidate ELSE 0 END) as dem_votes,
    SUM(CASE WHEN e.party='REPUBLICAN' THEN e.votes_candidate ELSE 0 END) as rep_votes,
    p.*,  -- All population columns
    u.*,
    ed.*
FROM election_data e
LEFT JOIN population_data p ON e.fips_code = p.fips_code
LEFT JOIN unemployment_data u ON e.fips_code = u.fips_code
LEFT JOIN education_data ed ON e.fips_code = ed.fips_code
WHERE e.year=2020
GROUP BY e.fips_code, e.year, e.state, e.county_name
"""

model_df = pd.read_sql_query(query, conn)
conn.close()

print("Model DataFrame (model_df) created successfully:")
print(model_df.head())

# Create a target variable: democratic vote share
model_df['dem_share'] = model_df['dem_votes'] / (model_df['dem_votes'] + model_df['rep_votes'])

# Drop identifying or non-numeric columns
id_cols = ['fips_code', 'year', 'state', 'county_name', 'dem_votes', 'rep_votes']
X = model_df.drop(columns=id_cols + ['dem_share'], errors='ignore')

# Keep only numeric columns
X = X.select_dtypes(include=[np.number]).copy()

# Drop rows with missing values
X = X.dropna()
y = model_df.loc[X.index, 'dem_share'].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Performance:")
print("MAE:", mae)
print("RMSE:", rmse)

importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print("Feature Importances:")
print(importances.head(10))