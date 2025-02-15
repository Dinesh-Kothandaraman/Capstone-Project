# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# from langchain.docstore.document import Document

# class DocGPT:
#     def __init__(self, docs, embedding_model="BAAI/bge-large-en"):
#         self.docs = docs
#         self.qa_chain = None
#         self.embedding_model = embedding_model
#         self._llm = None

#     def _preprocess_docs(self):
#         """Chunks documents to improve retrieval."""
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
#         chunked_docs = []

#         for doc in self.docs:
#             chunks = text_splitter.split_text(doc.page_content)
#             for chunk in chunks:
#                 chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

#         return chunked_docs

#     def _embeddings(self):
#         """Creates embeddings and vector database."""
#         embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
#         chunked_docs = self._preprocess_docs()
#         db = FAISS.from_documents(chunked_docs, embedding=embeddings)
#         return db

#     def create_qa_chain(self, retriever_k=5, model_path="D:/New folder"):
#         """Sets up the retrieval-augmented generation (RAG) pipeline."""
#         db = self._embeddings()
#         retriever = db.as_retriever(search_kwargs={"k": retriever_k})

#         # Load Local LLM
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path, device_map="auto", torch_dtype="auto"
#         )

#         # Create LLM pipeline
#         pipe = pipeline(
#             "text-generation", model=model, tokenizer=tokenizer,
#             max_new_tokens=100, temperature=0.7, do_sample=True
#         )
#         local_llm = HuggingFacePipeline(pipeline=pipe)

#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=local_llm,
#             retriever=retriever,
#             return_source_documents=True,  # Returns retrieved docs for debugging
#             verbose=True
#         )

#     def run(self, query: str) -> str:
#         """Processes user query and returns generated response."""
#         if not self.qa_chain:
#             return "QA chain not initialized. Please create the QA chain first."

#         response = self.qa_chain(query)
#         answer = response.get("result", "No answer generated.")

#         # Debug: Show retrieved docs
#         retrieved_docs = response.get("source_documents", [])
#         retrieved_texts = "\n\n".join([doc.page_content for doc in retrieved_docs])

#         return f"Question: {query}\n\nAnswer: {answer}\n\nRetrieved Context:\n{retrieved_texts}"

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
        self.docs = docs
        self.qa_chain = None
        self.embedding_model = embedding_model
        self._llm = None
        self._db = None  # Store the FAISS DB to avoid recomputation

    def _preprocess_docs(self):
        """Optimized document chunking."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
        chunked_docs = [
            Document(page_content=chunk, metadata=doc.metadata)
            for doc in self.docs
            for chunk in text_splitter.split_text(doc.page_content)
        ]
        return chunked_docs

    def _embeddings(self):
        """Creates embeddings and vector database with FAISS."""
        if self._db:
            return self._db  # Avoid redundant DB creation
        
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        chunked_docs = self._preprocess_docs()
        self._db = FAISS.from_documents(chunked_docs, embedding=embeddings)
        return self._db

    def create_qa_chain(self, retriever_k=3, model_path="D:/New folder"):
        """Sets up the optimized RAG pipeline."""
        db = self._embeddings()
        retriever = db.as_retriever(search_kwargs={"k": retriever_k})

        # Load Tokenizer and Model Efficiently
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16
        )

        # Create Optimized LLM Pipeline
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=80, temperature=0.6, do_sample=True
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm, retriever=retriever, return_source_documents=False, verbose=False
        )

    def run(self, query: str) -> str:
        """Processes user query and returns generated response efficiently."""
        if not self.qa_chain:
            return "QA chain not initialized. Please create the QA chain first."

        response = self.qa_chain(query)
        return response.get("result", "No answer generated.")
