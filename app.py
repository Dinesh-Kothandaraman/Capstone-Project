import streamlit as st
import pandas as pd
import requests
import sqlite3

# API Configuration
API_KEY = "D9AKJQG6VC99GUBR"
BASE_URL = "https://www.alphavantage.co/query"
PARAMS = {
    "function": "TIME_SERIES_MONTHLY",
    "symbol": "IBM",
    "apikey": API_KEY
}

DATABASE = "question_history.db"

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

def initialize_database():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS question_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_to_history(question):
    """Save a new question to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO question_history (question) VALUES (?)", (question,))
    conn.commit()
    conn.close()

def load_history():
    """Load the question history from the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT question FROM question_history")
    history = [row[0] for row in cursor.fetchall()]
    conn.close()
    return history

def main():
    st.title("Stock Data Viewer with Question Logging")

    # Initialize database
    initialize_database()

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

        # Input for user questions
        st.write("### Ask a Question")
        query = st.text_input("Enter your question here:")
        if query:
            save_to_history(query)
            st.success("Question saved!")

        # Display question history
        question_history = load_history()
        if question_history:
            st.write("### Question History")
            for i, question in enumerate(question_history, 1):
                st.write(f"{i}. {question}")

if __name__ == "__main__":
    main()
