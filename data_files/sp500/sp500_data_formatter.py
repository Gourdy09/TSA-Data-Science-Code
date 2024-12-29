import pandas as pd

with open('data_files/sp500/sp500_data.txt', 'r') as file:
    content = file.read()

data_array = content.split("|")

date_array = []
price_array = []

count = 0
for x in data_array:
    if(count % 2 == 0):
        date_array.append(x)
    else:
        price_array.append(x)
    count += 1

df = pd.DataFrame({
    'Date': date_array,
    'Value': price_array
})

df.to_csv('data_files/sp500/sp500_data.csv', index=False)

print("S&P500_data.csv created sucessfully")