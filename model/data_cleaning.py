import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import csv

election_data = pd.read_csv("countypres_2000-2020.csv")

# Inspecting columns and cleaning the Election Data
election_data_cleaned = election_data[
    ['year', 'state', 'state_po', 'county_name', 'county_fips', 'candidate',
     'party', 'candidatevotes', 'totalvotes']
].dropna()

# Renaming columns for clarity and consistency
election_data_cleaned = election_data_cleaned.rename(columns={
    'state_po': 'state_abbreviation',
    'county_fips': 'fips_code',
    'candidatevotes': 'votes_candidate',
    'totalvotes': 'votes_total'
})

# Analyzing election results for each county in each election year

# Group data by year, state, county, and party to calculate total votes by party
county_election_results = election_data_cleaned.groupby(
    ['year', 'state', 'county_name', 'party']
).agg({'votes_candidate': 'sum'}).reset_index()

# Find the winning party for each county in each year
county_election_results['rank'] = county_election_results.groupby(
    ['year', 'state', 'county_name']
)['votes_candidate'].rank(method='max', ascending=False)

# Filter to get only the winning party for each county-year
winning_party_per_county = county_election_results[county_election_results['rank'] == 1].drop(columns=['rank'])

# Sort results for better readability
winning_party_per_county = winning_party_per_county.sort_values(by=['year', 'state', 'county_name'])

# Identifying swing counties (counties that changed their winning party between elections)

# Create a DataFrame of winning parties by year, state, and county
winning_party_pivot = winning_party_per_county.pivot_table(
    index=['state', 'county_name'],
    columns='year',
    values='party',
    aggfunc='first'
).reset_index()

# Identify swing counties (counties with more than one unique winning party)
winning_party_pivot['unique_parties'] = winning_party_pivot.iloc[:, 2:].nunique(axis=1)
swing_counties = winning_party_pivot[winning_party_pivot['unique_parties'] > 1]

# Identify consistent counties (counties with the same winning party across all elections)
consistent_counties = winning_party_pivot[winning_party_pivot['unique_parties'] == 1]

# Drop the helper column `unique_parties` for presentation
swing_counties = swing_counties.drop(columns=['unique_parties'])
consistent_counties = consistent_counties.drop(columns=['unique_parties'])

#visualizing the swing counties and consistent counties per by state

# Count the number of swing and consistent counties per state
swing_counts_by_state = swing_counties.groupby('state').size().reset_index(name='swing_counties')
consistent_counts_by_state = consistent_counties.groupby('state').size().reset_index(name='consistent_counties')

# Merge the counts for visualization
state_counts = pd.merge(swing_counts_by_state, consistent_counts_by_state, on='state', how='outer').fillna(0)

# Visualization of swing and consistent counties by state
plt.figure(figsize=(12, 8))

# Bar chart for swing counties
plt.bar(state_counts['state'], state_counts['swing_counties'], label='Swing Counties')

# Bar chart for consistent counties stacked on top
plt.bar(state_counts['state'], state_counts['consistent_counties'], bottom=state_counts['swing_counties'], label='Consistent Counties')

# Add chart details
plt.xlabel('State')
plt.ylabel('Number of Counties')
plt.title('Swing and Consistent Counties by State')
plt.xticks(rotation=90)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

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

# Remove specified columns
columns_to_remove = [
    'Rural_Urban_Continuum_Code_2013',
    'Urban_Influence_Code_2013',
    'Metro_2013',
    'Median_Household_Income_2021',
    'Med_HH_Income_Percent_of_State_Total_2021'
]
unemployment_data_cleaned = unemployment_data_cleaned.drop(columns=columns_to_remove, errors='ignore')

