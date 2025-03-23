import streamlit as st
import requests
import sqlite3
import json
import PyPDF2  # For PDF processing
from langchain.schema import Document
from docGPT.docGPT import DocGPT
import base64
import matplotlib.pyplot as plt
import finnhub
import yfinance as yf
import pandas as pd

# API Configuration
# STOCK_API_KEY = "43YRXE6IDCMHN6W7"  # Replace with your real API Key
STOCK_API_KEY = "T5VEZRPRX8PBSGYB"
STOCK_BASE_URL = "https://www.alphavantage.co/query"

NEWS_API_KEY = "demo"  # Replace with your real News API Key
NEWS_API_URL = "https://www.alphavantage.co/query"

QA_PAIRS_PATH = r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\Data\news_qa_pairs.json"  # Pre-generated QA file
DATABASE = "question_history.db"


# Function to read uploaded document
def extract_text_from_file(uploaded_file):
    """Extracts text from uploaded .txt or .pdf files."""
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "txt":
            return uploaded_file.read().decode("utf-8")
        elif file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        else:
            st.error("Unsupported file format. Please upload a .txt or .pdf file.")
            return None
    return None

api_key = "0eca00d3c7242d98f65db1fa0782bbdc"
symbols = ['AAPL','NVDA','MSFT','GOOG','AMZN','META','BRK.B','V','MA','UNH','LLY','NVO','2222.SR','XOM','TSLA','AVGO','JPM','JNJ','WMT','PG','NESN.SW','ROG.SW','MC.PA','TM','005930.KS','NSRGY','PFE','CSCO','ORCL','INTC','GOOGL','IBM']
# symbols =['IBM','GOOGL']
base_url = "http://api.marketstack.com/v2/eod"

def fetch_stock_data(symbols, api_key, base_url):
    """Fetches real-time stock data from MarketStack API for given symbols and returns as Document objects."""
    all_stock_docs = []

    for symbol in symbols:
        params = {
            "access_key": api_key,
            "symbols": symbol
        }
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            stock_data = response.json().get("data", [])

            if not stock_data:
                st.warning(f"No stock data found for {symbol}.")
                continue

            for entry in stock_data:
                doc_content = (
                    f"Symbol: {symbol}\n"
                    f"Date: {entry['date']}\n"
                    f"Open: {entry['open']}\n"
                    f"High: {entry['high']}\n"
                    f"Low: {entry['low']}\n"
                    f"Close: {entry['close']}\n"
                    f"Volume: {entry['volume']}"
                )
                all_stock_docs.append(Document(page_content=doc_content))

        else:
            st.write(f"Stock API request failed for {symbol}: {response.status_code}")

    return all_stock_docs


# Fetch Stock Data from API
# def fetch_stock_data():
#     """Fetches real-time stock data from Alpha Vantage API."""
#     params = {
#         "function": "TIME_SERIES_MONTHLY",
#         "symbol": "IBM",
#         "apikey": STOCK_API_KEY
#     }
#     response = requests.get(STOCK_BASE_URL, params=params)

#     if response.status_code == 200:
#         stock_data = response.json()
#         time_series = stock_data.get("Monthly Time Series", {})

#         if not time_series:
#             st.warning("No stock data found.")
#             return []

#         return [
#             Document(page_content=f"Stock Date: {date}\nStock Data: {json.dumps(data)}")
#             for date, data in time_series.items()
#         ]
#     else:
#         st.error(f"Stock API request failed: {response.status_code}")
#         return []

# Fetch Stock-Related News from API
def fetch_stock_news():
    """Fetches real-time stock-related news from Alpha Vantage API."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "AAPL",
        "apikey": NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)

    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("feed", [])

        if not articles:
            st.warning("No relevant news found.")
            return []

        return [
            Document(page_content=f"News Title: {article['title']}\nSummary: {article['summary']}\nSentiment: {article['overall_sentiment_label']}")  
            for article in articles
        ]
    else:
        st.error(f"News API request failed: {response.status_code}")
        return []

symbols_list = ["AAPL", "GOOGL", "MSFT", "TSLA","NVDA","IBM"]
finnhub_client = finnhub.Client(api_key="cve8bt9r01ql1jn9q620cve8bt9r01ql1jn9q62g")

def fetch_multiple_stock_news(symbols, start_date="2025-01-01", end_date="2025-03-20"):
    """Fetches stock-related news for multiple companies from Finnhub API."""
    all_news = []

    for symbol in symbols:
        news_articles = finnhub_client.company_news(symbol, _from=start_date, to=end_date)

        if not news_articles:
            st.warning(f"No relevant news found for {symbol}.")
            continue

        all_news.extend([
            Document(
                page_content=f"Stock Symbol: {symbol}\nNews Title: {article['headline']}\nSummary: {article['summary']}\nSource: {article['source']}\nURL: {article['url']}"
            )
            for article in news_articles
        ])

    return all_news

symbols1 = ["AAPL" , "GOOGL", "MSFT","TSLA","NVDA","IBM"]

# symbols = ['AAPL','NVDA','MSFT','GOOG','AMZN','META','BRK.B','V','MA','UNH','LLY','NVO','2222.SR','XOM','TSLA','AVGO','JPM','JNJ','WMT','PG','NESN.SW','ROG.SW','MC.PA','TM','005930.KS','NSRGY','PFE','CSCO','ORCL','INTC','GOOGL','IBM']

def fetch_yfinance_data(symbols):
    """Fetches historical stock data using yfinance and returns it as a list of Document objects."""
    all_stock_docs = []
    
    # Fetch stock data for all companies at once
    df = yf.download(symbols, group_by='ticker')
    
    # Loop through each stock symbol
    for symbol in symbols:
        if symbol not in df:
            st.warning(f"No data found for {symbol}.")
            continue
        
        stock_df = df[symbol]  # Extract data for the specific stock
        
        # Loop through the DataFrame to create Document objects
        for date, row in stock_df.iterrows():
            date_str = date.strftime('%Y-%m-%d')  # Format date
            doc_content = (
                f"Symbol: {symbol}\n"
                f"Date: {date_str}\n"
                f"Open: {row['Open']}\n"
                f"High: {row['High']}\n"
                f"Low: {row['Low']}\n"
                f"Close: {row['Close']}\n"
                f"Volume: {row['Volume']}"
            )
            
            all_stock_docs.append(Document(page_content=doc_content))
    
    return all_stock_docs

# def fetch_stock_news1():
#     """Reads stock news from a CSV file and returns it as a list of Document objects."""
#     all_news_docs = []
    
#     # Read CSV file
#     df = pd.read_csv(r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\Data\analyst_ratings_processed.csv")
    
#     # Loop through each row to create Document objects
#     for _, row in df.iterrows():
#         date_str = row['Date']  # Treat date as string
#         doc_content = (
#             f"Date: {date_str}\n"
#             f"Stock News: {row['Stock News']}"
#         )
        
#         all_news_docs.append(Document(page_content=doc_content))
    
#     return all_news_docs

# Load Pre-generated QA Pairs
def load_qa_pairs():
    """Loads pre-generated QA pairs from the JSON file."""
    try:
        with open(QA_PAIRS_PATH, "r") as f:
            qa_data = json.load(f)
        
        return [Document(page_content=f"Q: {item['question']} A: {item['answer']}") for item in qa_data]
    except Exception as e:
        st.error(f"Error loading QA pairs: {e}")
        return []

# Initialize Database
def initialize_database():
    """Creates a database table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS question_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Load Question History from Database
def load_history():
    """Retrieves past user queries from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM question_history")
    history = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()
    return history

# Save User Query and Response to Database
def save_to_history(question, answer):
    """Stores the user's query and the model's response."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO question_history (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

# def execute_code(code):
#     """Safely executes the generated Python visualization code."""
#     try:
#         exec(code, globals())
#     except Exception as e:
#         st.error(f"Error executing code: {e}")

def execute_code(code_str):
    """Safely executes the extracted Python code."""
    if not isinstance(code_str, str) or not code_str.strip():
        return {"error": "No valid code provided for execution."}

    try:
        exec(code_str, globals())
        return {"success": "Code executed successfully."}
    except Exception as e:
        return {"error": f"Error executing code: {e}"}

# Main Streamlit App
def main():
    st.title("📊 Stock & News QA System (Two APIs + QA Pairs + Optional Documents)")

    # Initialize database
    initialize_database()
    question_history = load_history()

    # Step 1: Fetch stock and news data
    st.write("🔄 Fetching stock data and news...")
    stock_docs = fetch_stock_data(symbols, api_key, base_url)
    st.write("stock_data",len(stock_docs))
    # st.write("stock_data",stock_docs)
    news_docs = fetch_stock_news()
    st.write("news_data",len(news_docs))
    c = fetch_multiple_stock_news(symbols_list, start_date="2025-01-01", end_date="2025-03-20")
    st.write("c",len(c))
    d = fetch_yfinance_data(symbols1)
    st.write("d",len(d))
    # e = fetch_stock_news1()
    # st.write("e",len(e))

    # st.write("stock_news",news_docs)

    # Step 2: Load pre-generated QA pairs
    st.write("🔄 Loading QA pairs dataset...")
    qa_docs = load_qa_pairs()

    # Step 3: Upload a document (Optional)
    st.subheader("📂 Upload a Document (TXT or PDF) - *Optional*")
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file (or skip)", type=["txt", "pdf"])

    uploaded_docs = []
    if uploaded_file:
        extracted_text = extract_text_from_file(uploaded_file)
        if extracted_text:
            uploaded_docs.append(Document(page_content=extracted_text))
            st.success("✅ File uploaded and processed successfully!")
        else:
            st.warning("⚠️ File processing failed. Proceeding without a document.")
    else:
        st.info("ℹ️ No document uploaded. Skipping file processing.")

    # Step 4: Process Data Only After File Upload or Skip
    if "doc_gpt" not in st.session_state:
        st.session_state.doc_gpt = None  # Ensure it starts as None

    if st.button("🚀 Process Data and Train Model"):
        docs = stock_docs + news_docs + c  + qa_docs + uploaded_docs
        # docs = qa_docs
        st.write(len(docs))
        
        if not docs:
            st.error("No data available for training.")
            return
        
        # st.session_state.docs = docs

        # Train the DocGPT model with API stock data, news articles, and QA pairs
        st.write("🧠 Training model with API data + QA pairs...")
        st.session_state.doc_gpt = DocGPT(docs)
        st.write("Processing data...")
        st.session_state.doc_gpt.create_qa_chain()
        st.success("✅ Training complete!")

    # Step 5: Accept user queries (only if model is trained)
    if st.session_state.doc_gpt is not None:
        query = st.text_input("🔍 Ask a question about stock data, news, or QA pairs:")

        if query:
            response = st.session_state.doc_gpt.run(query)
            if isinstance(response, dict) and "image" in response:
                display_image(response["image"])
                st.write("**Answer:**", response.get("result", "No answer generated."))
            else:
                save_to_history(query, response)
                question_history.append((query, response))
                st.write("**Answer:**", response)

        # if query:
        #     # response = st.session_state.doc_gpt.run(docs,query)
        #     response = st.session_state.doc_gpt.run(st.session_state.docs, query)


        #     if isinstance(response, dict) and "code" in response:
        #         st.subheader("📝 Generated Code:")
        #         st.code(response["code"], language="python")
        #         execute_code(response["code"])  # Run and display the visualization
        #     else:
        #         st.subheader("📖 Answer:")
        #         save_to_history(query, response)
        #         question_history.append((query, response))                
        #         st.write(response)

    # Display question history
    if question_history:
        st.write("### 📜 Question History")
        for i, (q, a) in enumerate(question_history, 1):
            st.write(f"{i}. **Q:** {q}  \n   **A:** {a}")


if __name__ == "__main__":
    main()
