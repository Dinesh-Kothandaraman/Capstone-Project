from fastapi import FastAPI
from src.inference import run_inference
from docGPT.docGPT import DocGPT

app = FastAPI()
doc_gpt = DocGPT([])  # Initialize with preloaded model/data

@app.post("/predict/")
def predict(query: str):
    response = run_inference(query, doc_gpt)
    return {"answer": response}