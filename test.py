import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=D9AKJQG6VC99GUBR'
r = requests.get(url)
data = r.json()

print(data)
#test

#D9AKJQG6VC99GUBR