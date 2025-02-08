import requests

# Replace with your actual API key and endpoint
API_KEY = "D9AKJQG6VC99GUBR"
BASE_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=IBM&apikey=D9AKJQG6VC99GUBR'

# Define parameters (example: fetching stock price for AAPL)
params = {
    "symbol": "AAPL",
    "interval": "1d",
    "apikey": API_KEY
}

# Make the API request
response = requests.get(BASE_URL, params=params)

# Check if request was successful
if response.status_code == 200:
    stock_data = response.json()
    # print(stock_data)  # Print or save the data
else:
    print("Error:", response.status_code, response.text)


import pandas as pd

# Convert JSON response to DataFrame (modify based on API response structure)
df = pd.DataFrame(stock_data["data"])

# Save to CSV
df.to_csv("stock_data.csv", index=False)
print("Stock data saved successfully!")
