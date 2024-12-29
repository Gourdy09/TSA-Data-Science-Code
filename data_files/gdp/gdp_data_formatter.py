import pandas as pd

# Load the HPI and CPI data
gdp_df = pd.read_csv('data_files/gdp/gdp.csv')
cpi_df = pd.read_csv('data_files/cpi/cpi_data.csv')

# Remove any extra spaces in column names (if any)
gdp_df.columns = gdp_df.columns.str.strip()
cpi_df.columns = cpi_df.columns.str.strip()

# Extract year from the 'Date' column of HPI data (gdp.csv has 'YYYY-MM-DD' format)
gdp_df['Year'] = pd.to_datetime(gdp_df['Date'], format='%Y-%m-%d').dt.year

# Extract year from the 'Date' column of CPI data (cpi.csv has 'MM/DD/YYYY' format)
cpi_df['Year'] = pd.to_datetime(cpi_df['Date'], format='%m/%d/%Y').dt.year

# Merge HPI data with CPI data on the 'Year' column, with custom suffixes
merged_df = pd.merge(gdp_df, cpi_df, on='Year', suffixes=('_gdp', '_cpi'))

# Calculate the nominal value for HPI using CPI for the corresponding year
merged_df['Value'] = merged_df['Value_gdp'] * (merged_df['Value_cpi'] / 100)

# Reformat the 'Date' column to MM/DD/YYYY (from 'Date_gdp')
merged_df['Date'] = pd.to_datetime(merged_df['Date_gdp'], format='%Y-%m-%d').dt.strftime('%m/%d/%Y')

# Final DataFrame with 'Date' and 'Value' columns
final_df = merged_df[['Date', 'Value']]

# Save to CSV with the new file name
final_df.to_csv('data_files/gdp/gdp_data.csv', index=False)

print("Nominal GDP data saved to gdp_data.csv")