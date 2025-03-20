import pandas as pd

# Load the HPI and CPI data
hpi_df = pd.read_csv('data_files/hpi/hpi.csv')
cpi_df = pd.read_csv('data_files/cpi/cpi_data.csv')

# Extract year from the 'Date' column of HPI data
hpi_df['Year'] = pd.to_datetime(hpi_df['Date'], format='%m/%d/%Y').dt.year

# Extract year from the 'Date' column of CPI data (you can just keep the 'Year' and 'Value')
cpi_df['Year'] = pd.to_datetime(cpi_df['Date'], format='%m/%d/%Y').dt.year

# Merge HPI data with CPI data on the 'Year' column
merged_df = pd.merge(hpi_df, cpi_df, on='Year', suffixes=('_hpi', '_cpi'))

# Calculate the nominal value for HPI using CPI for the corresponding year
merged_df['Value'] = merged_df['Value_hpi'] * (merged_df['Value_cpi'] / 100)

# Reformat the 'Date' column to MM/DD/YYYY
merged_df['Date'] = pd.to_datetime(merged_df['Date_hpi'], format='%m/%d/%Y').dt.strftime('%m/%d/%Y')

# Final DataFrame with 'Date' and 'Value' columns
final_df = merged_df[['Date', 'Value']]

# Save to CSV with the new file name
final_df.to_csv('data_files/hpi/hpi_data.csv', index=False)

print("Nominal HPI data saved to hpi_data.csv")


