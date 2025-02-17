from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.docstore.document import Document
import torch

class DocGPT:
    def __init__(self, docs, embedding_model="BAAI/bge-large-en"):
        """
        Initializes DocGPT with a given dataset.
        Supports multiple document types: stock data, news articles, QA pairs.
        """
        self.docs = docs
        self.qa_chain = None
        self.embedding_model = embedding_model
        self._llm = None
        self._db = None  # Store FAISS DB to avoid recomputation

    def _preprocess_docs(self):
        """Optimized document chunking to improve retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunked_docs = [
            Document(page_content=chunk, metadata=doc.metadata)
            for doc in self.docs
            for chunk in text_splitter.split_text(doc.page_content)
        ]
        return chunked_docs

    def _embeddings(self):
        """Creates embeddings and a FAISS vector database."""
        if self._db:
            return self._db  # Avoid redundant FAISS creation
        
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        chunked_docs = self._preprocess_docs()
        if not chunked_docs:
            raise ValueError("No documents available for embedding. Check data sources.")
        self._db = FAISS.from_documents(chunked_docs, embedding=embeddings)
        return self._db

    def create_qa_chain(self, retriever_k=5, model_path="D:/New folder"):
        """Sets up the RAG pipeline with an efficient retriever."""
        db = self._embeddings()
        retriever = db.as_retriever(search_kwargs={"k": retriever_k})

        # Load Tokenizer and Model Efficiently
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        )  #, device_map="auto"

        # Create Optimized LLM Pipeline
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=150, temperature=0.7, do_sample=True
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm, retriever=retriever, return_source_documents=False, verbose=False
        )

    def run(self, query: str) -> str:
        """Processes user query and returns a generated response efficiently."""
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please create the QA chain first."

        response = self.qa_chain(query)
        return response.get("result", "No answer generated.")
