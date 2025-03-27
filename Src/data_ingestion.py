import requests
from langchain.schema import Document

def fetch_stock_data(symbols, api_key, base_url):
    all_stock_docs = []
    for symbol in symbols:
        params = {"access_key": api_key, "symbols": symbol}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            stock_data = response.json().get("data", [])
            for entry in stock_data:
                doc_content = f"Symbol: {symbol}\nDate: {entry['date']}\nClose: {entry['close']}"
                all_stock_docs.append(Document(page_content=doc_content))
    return all_stock_docs