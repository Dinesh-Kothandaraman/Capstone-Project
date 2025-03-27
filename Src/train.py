from docGPT.docGPT import DocGPT
from src.preprocess import create_embeddings

def train_model(docs, model_path="path/to/model"):
    doc_gpt = DocGPT(docs)
    db = create_embeddings(docs)
    doc_gpt._db = db
    doc_gpt.create_qa_chain(model_path=model_path)
    doc_gpt.qa_chain.llm.pipeline.model.save_pretrained("models/trained_model")
    return doc_gpt