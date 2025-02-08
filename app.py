import streamlit as st
import pandas as pd
import requests

# API Configuration
API_KEY = "D9AKJQG6VC99GUBR"
BASE_URL = "https://www.alphavantage.co/query"
PARAMS = {
    "function": "TIME_SERIES_MONTHLY",
    "symbol": "IBM",
    "apikey": API_KEY
}

def fetch_api_data():
    """Fetches stock data from API and returns a Pandas DataFrame."""
    response = requests.get(BASE_URL, params=PARAMS)

    if response.status_code == 200:
        stock_data = response.json()
        time_series = stock_data.get("Monthly Time Series", {})

        if not time_series:
            st.error("Invalid API response. Check API key and parameters.")
            return None

        # Convert JSON to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index.name = "Date"
        df = df.rename(columns=lambda x: x[3:])  # Remove prefix from column names
        return df
    else:
        st.error(f"API request failed: {response.status_code} - {response.text}")
        return None

def main():
    st.title("Stock Data Viewer")

    # Fetch stock data from API
    st.write("Fetching stock data from API...")
    df = fetch_api_data()

    if df is not None:
        st.success("Data Loaded Successfully!")
        st.write("### Stock Data Overview")
        st.dataframe(df)  # Display data in a table

        # Show a chart of closing prices
        st.write("### Closing Price Trend")
        st.line_chart(df["close"].astype(float))  # Ensure data type is float for charting

if __name__ == "__main__":
    main()
