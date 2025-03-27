from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def preprocess_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return [Document(page_content=chunk, metadata=doc.metadata) 
            for doc in docs for chunk in text_splitter.split_text(doc.page_content)]

def create_embeddings(docs, embedding_model="BAAI/bge-large-en"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.from_documents(docs, embeddings)