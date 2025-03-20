import pandas as pd

with open('data_files/cpi/cpi_data.txt', 'r') as file:
    content = file.read()

data_lines = content.splitlines()


date_array = []
cpi_array = []


for line in data_lines:
    values = line.split(",")
    date = values[0]

    monthly_cpi_values = list(map(float, values[1:]))
    yearly_cpi = sum(monthly_cpi_values) / len(monthly_cpi_values)
    

    date_array.append(date)
    cpi_array.append(yearly_cpi)

df = pd.DataFrame({
    'Date': date_array,
    'Value': cpi_array
})

df.to_csv('data_files/cpi/cpi_data.csv', index=False)

print("cpi_data.csv created successfully")
