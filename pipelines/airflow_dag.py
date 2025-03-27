from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.data_ingestion import fetch_stock_data
from src.preprocess import preprocess_docs, create_embeddings
from src.train import train_model
import json
import os
from langchain.schema import Document


# Configuration (replace with your actual values)
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "IBM"]
api_key = "0eca00d3c7242d98f65db1fa0782bbdc"  # Replace with your actual MarketStack API key
base_url = "http://api.marketstack.com/v2/eod"
data_dir = "data/raw"
processed_dir = "data/processed"

# Task 1: Fetch data and save it
def fetch_and_save_data(**kwargs):
    """Fetch stock data and save to data/raw/."""
    stock_docs = fetch_stock_data(symbols, api_key, base_url)
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "stock_data.json")
    with open(data_path, "w") as f:
        json.dump([doc.page_content for doc in stock_docs], f)  # Save as JSON
    return data_path  # Return the file path for downstream tasks

# Task 2: Preprocess data and save embeddings
def preprocess_and_embed(**kwargs):
    """Preprocess data and save embeddings."""
    ti = kwargs["ti"]  # Task instance to pull data from previous task
    data_path = ti.xcom_pull(task_ids="fetch_data")  # Get the file path from fetch task
    with open(data_path, "r") as f:
        raw_data = [Document(page_content=chunk) for chunk in json.load(f)]
    processed_docs = preprocess_docs(raw_data)
    db = create_embeddings(processed_docs)
    embeddings_path = os.path.join(processed_dir, "faiss_index")
    os.makedirs(processed_dir, exist_ok=True)
    db.save_local(embeddings_path)
    return embeddings_path

# Task 3: Train the model
def train(**kwargs):
    """Train the model using preprocessed data."""
    ti = kwargs["ti"]
    data_path = ti.xcom_pull(task_ids="fetch_data")  # Reuse raw data
    with open(data_path, "r") as f:
        raw_data = [Document(page_content=chunk) for chunk in json.load(f)]
    model_path = r"C:\Users\dines\Downloads\Capstone_model"
    train_model(raw_data, model_path=model_path)
    return model_path

# Define the DAG
default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "docgpt_pipeline",
    start_date=datetime(2025, 3, 27),
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:
    fetch_task = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_and_save_data,
        provide_context=True,  # Allows access to kwargs like ti
    )
    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_and_embed,
        provide_context=True,
    )
    train_task = PythonOperator(
        task_id="train",
        python_callable=train,
        provide_context=True,
    )

    # Set task dependencies
    fetch_task >> preprocess_task >> train_task