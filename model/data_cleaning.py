import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import csv
import seaborn as sns

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

# File paths
file_2000_2010 = 'population-2000-2010.csv'
file_2010_2020 = 'population-2010-2020.csv'
file_2020_2023 = 'population-2020-2023.csv'

# Read the CSV files
pop_2000_2010 = pd.read_csv(file_2000_2010, encoding='ISO-8859-1')
pop_2010_2020 = pd.read_csv(file_2010_2020, encoding='ISO-8859-1')
pop_2020_2023 = pd.read_csv(file_2020_2023, encoding='ISO-8859-1')

# Define columns to keep for each time range
columns_to_keep_2000 = ['STATE', 'COUNTY', 'STNAME'] + [f'POPESTIMATE{year}' for year in range(2000, 2011)]
columns_to_keep_2010 = ['STATE', 'COUNTY', 'STNAME'] + [f'POPESTIMATE{year}' for year in range(2011, 2021)]  # Exclude 2010
columns_to_keep_2020 = ['STATE', 'COUNTY', 'STNAME'] + [f'POPESTIMATE{year}' for year in range(2021, 2023)]  # Exclude 2020

# Filter the columns for each dataset
pop_2000_2010_filtered = pop_2000_2010[columns_to_keep_2000]
pop_2010_2020_filtered = pop_2010_2020[columns_to_keep_2010]
pop_2020_2023_filtered = pop_2020_2023[columns_to_keep_2020]

# Reset index before merging
pop_2000_2010_filtered = pop_2000_2010_filtered.reset_index(drop=True)
pop_2010_2020_filtered = pop_2010_2020_filtered.reset_index(drop=True)
pop_2020_2023_filtered = pop_2020_2023_filtered.reset_index(drop=True)

# Merge 2000-2010 with 2010-2020
combined_df = pd.merge(
    pop_2000_2010_filtered,
    pop_2010_2020_filtered,
    on=['STATE', 'COUNTY', 'STNAME'],
    how='inner'  # Change to 'outer' if you want to retain all rows
)

# Merge the above result with 2020-2023
combined_df = pd.merge(
    combined_df,
    pop_2020_2023_filtered,
    on=['STATE', 'COUNTY', 'STNAME'],
    how='inner'  # Change to 'outer' if you want to retain all rows
)

# Optional: Sort the DataFrame by STATE and COUNTY
combined_df = combined_df.sort_values(by=['STATE', 'COUNTY']).reset_index(drop=True)

# Display the combined DataFrame
combined_df.head()
# Save as a CSV file
output_file = 'combined_population_data.csv'
combined_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"DataFrame saved as {output_file}")

# Assuming combined_df is your merged DataFrame
# Ensure that all POPESTIMATE columns are numeric
pop_columns = [col for col in combined_df.columns if 'POPESTIMATE' in col]
combined_df[pop_columns] = combined_df[pop_columns].apply(pd.to_numeric, errors='coerce')

# Calculate total population per year
total_population = combined_df[pop_columns].sum()

# Extract years from column names
years = [int(col.replace('POPESTIMATE', '')) for col in total_population.index]
pop_values = total_population.values

# Create a DataFrame for plotting
total_pop_df = pd.DataFrame({
    'Year': years,
    'Total Population': pop_values
})

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=total_pop_df, x='Year', y='Total Population', marker='o')
plt.title('Total US Population Over Time (2000-2022)')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.tight_layout()
plt.show()

# Filepath to the uploaded Excel file
file_path = "Education.xlsx"

# Load the Excel file to inspect sheet names
excel_file = pd.ExcelFile(file_path)

# Display all sheet names
sheet_names = excel_file.sheet_names
print("Sheet names in the Excel file:", sheet_names)

# Define a function to clean each sheet
def clean_sheet(education_sheet_data):
    """Clean and preprocess a sheet's DataFrame."""
    # Drop rows or columns with all NaN values
    education_sheet_data = education_sheet_data.dropna(how='all').dropna(axis=1, how='all')

    # Reset index for consistency
    education_sheet_data = education_sheet_data.reset_index(drop=True)

    # Ensure columns are well-named
    education_sheet_data.columns = [col.strip() if isinstance(col, str) else col for col in education_sheet_data.columns]

    return education_sheet_data

# Initialize a dictionary to store cleaned data
cleaned_data = {}

# Iterate over each sheet and clean the data
for sheet in sheet_names:
    print(f"Cleaning sheet: {sheet}")
    sheet_data = excel_file.parse(sheet)
    cleaned_data[sheet] = clean_sheet(sheet_data)

# Example: Display the first few rows of each cleaned sheet
for sheet, data in cleaned_data.items():
    print(f"\nFirst rows of cleaned sheet '{sheet}':")
    print(data.head())

