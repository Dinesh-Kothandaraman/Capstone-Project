from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

class DocGPT:
    def __init__(self, docs):
        self.docs = docs
        self.qa_chain = None
        self._llm = None

    def _embeddings(self):
        # Use HuggingFace SentenceTransformers for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(self.docs, embedding=embeddings)
        return db

    def create_qa_chain(self, chain_type="stuff", verbose=True):
        db = self._embeddings()
        retriever = db.as_retriever()

        # Use HuggingFace Pipeline for local LLM
        # local_llm = HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="google/flan-t5-small"))
        # model_name = "facebook/opt-6.7b"  # Choose a model size like opt-2.7b, opt-6.7b, or opt-13b.
        model_name = r"D:\New folder"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")  #,offload_folder="offload")
        # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=25)
        local_llm = HuggingFacePipeline(pipeline=pipe)
        self.qa_chain = RetrievalQA.from_chain_type(llm=local_llm, retriever=retriever, verbose=False)


    def run(self, query: str) -> str:
        if not self.qa_chain:
            return "QA chain not initialized. Please create the QA chain first."

        # Get the answer
        answer = self.qa_chain.run(query)

        # Format as "Question" and "Answer"
        return f"Question: {query}\nAnswer: {answer}"


# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, pipeline
# from langchain_community.llms import HuggingFacePipeline
# import faiss


# class DocGPT:
#     def __init__(self, docs):
#         self.docs = docs
#         self.qa_chain = None
#         self._llm = None

#     # def _embeddings(self):
#     #     """Load FAISS index with precomputed embeddings."""
#     #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     #     # db = FAISS.load_local("faiss_index.bin", embeddings)
#     #     # db = FAISS.load_local("faiss_index.bin", embeddings, allow_dangerous_deserialization=True)
#     #     db = FAISS.load_local(r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\faiss_index.bin", embeddings, allow_dangerous_deserialization=True)
#     #     return db

#     def _embeddings(self):
#         # Initialize the HuggingFace embeddings
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
#         # Manually load the FAISS index
#         faiss_index_path = r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\faiss_index.bin"
        
#         try:
#             # Load the index using faiss.read_index() directly
#             index = faiss.read_index(faiss_index_path)
#         except Exception as e:
#             print(f"Error loading FAISS index: {e}")
#             return None
        
#         # Create a docstore (you may replace this with your actual document store if needed)
#         docstore = {}
        
#         # Create a mapping between the index and the docstore
#         index_to_docstore_id = {}
        
#         try:
#             # Wrap the index with FAISS vector store
#             db = FAISS(index, docstore, index_to_docstore_id, embeddings)
#             print("FAISS vector store initialized successfully.")
#         except Exception as e:
#             print(f"Error initializing FAISS vector store: {e}")
#             return None
        
#         return db

#     def create_qa_chain(self, verbose=True):
#         """Initialize RAG model with FAISS retrieval."""
#         db = self._embeddings()
#         retriever = db.as_retriever()

#         # Load fine-tuned RAG model and tokenizer
#         model_name = r"D:\New folder"  # Path to fine-tuned RAG model
#         tokenizer = RagTokenizer.from_pretrained(model_name)
#         retriever = RagRetriever.from_pretrained(model_name, index_name="custom", use_dummy_dataset=True)
#         retriever.index.deserialize_from("faiss_index.bin")

#         model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever)

#         # Set up text generation pipeline
#         pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
#         local_llm = HuggingFacePipeline(pipeline=pipe)

#         self.qa_chain = RetrievalQA.from_chain_type(llm=local_llm, retriever=retriever, verbose=False)

#     def run(self, query: str) -> str:
#         """Run RAG-based question answering."""
#         if not self.qa_chain:
#             return "QA chain not initialized. Please create the QA chain first."

#         answer = self.qa_chain.run(query)
#         return f"Question: {query}\nAnswer: {answer}"
