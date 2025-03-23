import json
from docGPT import DocGPT
import requests
from langchain.schema import Document  # Import Document class

# API Configuration
STOCK_API_KEY = "43YRXE6IDCMHN6W7"  # Replace with your real API Key
STOCK_BASE_URL = "https://www.alphavantage.co/query"

NEWS_API_KEY = "43YRXE6IDCMHN6W7"  # Replace with your real News API Key
NEWS_API_URL = "https://www.alphavantage.co/query"

# Fetch Stock Data from API
def fetch_stock_data():
    """Fetches real-time stock data from Alpha Vantage API."""
    params = {
        "function": "TIME_SERIES_MONTHLY",
        "symbol": "IBM",
        "apikey": STOCK_API_KEY
    }
    response = requests.get(STOCK_BASE_URL, params=params)

    if response.status_code == 200:
        stock_data = response.json()
        time_series = stock_data.get("Monthly Time Series", {})

        if not time_series:
            return []

        return [
            Document(page_content=f"Stock Date: {date}\nStock Data: {json.dumps(data)}")
            for date, data in time_series.items()
        ]
    else:
        return []

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
            return []

        return [
            Document(page_content=f"News Title: {article['title']}\nSummary: {article['summary']}\nSentiment: {article['overall_sentiment_label']}")  
            for article in articles
        ]
    else:
        return []


# Path to your evaluation dataset (Update this path)
EVAL_DATASET_PATH = r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\Data\news_qa_pairs.json"

def load_eval_dataset():
    """Loads evaluation dataset containing questions and expected answers."""
    try:
        with open(EVAL_DATASET_PATH, "r") as f:
            eval_data = json.load(f)
        return eval_data
    except Exception as e:
        print(f"Error loading evaluation dataset: {e}")
        return []

def evaluate_model(doc_gpt, eval_data):
    """Evaluates the model on the evaluation dataset."""
    if not eval_data:
        print("No evaluation data found. Exiting...")
        return

    correct = 0
    total = len(eval_data)
    
    print("\n=== Model Evaluation ===\n")
    for idx, item in enumerate(eval_data, start=1):
        question = item.get("question", "").strip()
        expected_answer = item.get("answer", "").strip()

        if not question or not expected_answer:
            print(f"Skipping invalid entry at index {idx}")
            continue
        
        generated_answer = doc_gpt.run(question).strip()

        print(f"Q{idx}: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Generated: {generated_answer}\n")

        if generated_answer.lower() == expected_answer.lower():
            correct += 1

    accuracy = (correct / total) * 100
    print(f"\n=== Evaluation Complete ===")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Model Accuracy: {accuracy:.2f}%")

# Step 1: Load evaluation dataset
eval_data = load_eval_dataset()
qadocs = [Document(page_content=f"Q: {item['question']} A: {item['answer']}") for item in eval_data]

stock_docs = fetch_stock_data()
news_docs = fetch_stock_news()
qa_docs = qadocs

docs = stock_docs + news_docs + qa_docs

if not docs:
    print("Error: No documents available for embedding. Ensure your evaluation dataset is not empty.")
    exit(1)  # Stop execution

# Step 2: Initialize DocGPT model (Ensure you have trained it)
doc_gpt = DocGPT(docs)
doc_gpt.create_qa_chain()

# Step 3: Run Evaluation
evaluate_model(doc_gpt, eval_data)
