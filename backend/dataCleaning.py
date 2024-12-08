import pandas as pd
import sqlite3
import csv

election_data = pd.read_csv("countypres_2000-2020.csv")

election_data_cleaned = election_data[
    ['year', 'state', 'state_po', 'county_name', 'county_fips', 'candidate',
     'party', 'candidatevotes', 'totalvotes']
].dropna()

election_data_cleaned = election_data_cleaned.rename(columns={
    'state_po': 'state_abbreviation',
    'county_fips': 'fips_code',
    'candidatevotes': 'votes_candidate',
    'totalvotes': 'votes_total'
})

# Create a SQLite database and establish a connection
db_path = "election_data.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Define the schema for the election data table
create_table_query = """
CREATE TABLE IF NOT EXISTS election_data (
    year INTEGER,
    state TEXT,
    state_abbreviation TEXT,
    county_name TEXT,
    fips_code INTEGER,
    candidate TEXT,
    party TEXT,
    votes_candidate INTEGER,
    votes_total INTEGER
);
"""
cursor.execute(create_table_query)

# Insert cleaned data into the database
election_data_cleaned.to_sql("election_data", conn, if_exists="replace", index=False)

# Verify that the data has been inserted successfully
row_count = cursor.execute("SELECT COUNT(*) FROM election_data").fetchone()[0]

# Commit changes and close the connection
conn.commit()
conn.close()

unemployment_data = pd.read_csv("Unemployment(UnemploymentMedianIncome).csv")
list_of_column_names = []

with open('Unemployment(UnemploymentMedianIncome).csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    dict_from_csv = dict(list(csv_reader)[0])

    list_of_column_names = list(dict_from_csv.keys())

unemployment_data_cleaned = unemployment_data[list_of_column_names].dropna()

columns = []
for i in range(0, 22):
    new_str = 'employed_' + str(2000 + i)
    new_str2 = 'unemployed_' + str(2000 + i)
    new_str3 = 'unemployment_rate_' + str(2000 + i)
    columns.append(new_str)
    columns.append(new_str2)
    columns.append(new_str3)

new_columns = ["fips_code", "state", "county_name", "rural_code", "urban_code", "metro", "civilian_labor_force",
               columns]

poverty_data_cleaned.columns = new_columns

