# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM

# class DocGPT:
#     def __init__(self, docs):
#         self.docs = docs
#         self.qa_chain = None
#         self._llm = None

#     def _embeddings(self):
#         # Use HuggingFace SentenceTransformers for embeddings
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         db = FAISS.from_documents(self.docs, embedding=embeddings)
#         return db

#     def create_qa_chain(self, chain_type="stuff", verbose=True):
#         db = self._embeddings()
#         retriever = db.as_retriever()

#         # Use HuggingFace Pipeline for local LLM
#         # local_llm = HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="google/flan-t5-small"))
#         # model_name = "facebook/opt-6.7b"  # Choose a model size like opt-2.7b, opt-6.7b, or opt-13b.
#         model_name = r"D:\New folder"

#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")  #,offload_folder="offload")
#         # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
#         pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=25)
#         local_llm = HuggingFacePipeline(pipeline=pipe)
#         self.qa_chain = RetrievalQA.from_chain_type(llm=local_llm, retriever=retriever, verbose=False)


#     def run(self, query: str) -> str:
#         if not self.qa_chain:
#             return "QA chain not initialized. Please create the QA chain first."

#         # Get the answer
#         answer = self.qa_chain.run(query)

#         # Format as "Question" and "Answer"
#         return f"Question: {query}\nAnswer: {answer}"
    


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.docstore.document import Document

class DocGPT:
    def __init__(self, docs, embedding_model="BAAI/bge-large-en"):
        self.docs = docs
        self.qa_chain = None
        self.embedding_model = embedding_model
        self._llm = None

    def _preprocess_docs(self):
        """Chunks documents to improve retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunked_docs = []

        for doc in self.docs:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

        return chunked_docs

    def _embeddings(self):
        """Creates embeddings and vector database."""
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        chunked_docs = self._preprocess_docs()
        db = FAISS.from_documents(chunked_docs, embedding=embeddings)
        return db

    def create_qa_chain(self, retriever_k=5, model_path="D:/New folder"):
        """Sets up the retrieval-augmented generation (RAG) pipeline."""
        db = self._embeddings()
        retriever = db.as_retriever(search_kwargs={"k": retriever_k})

        # Load Local LLM
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype="auto"
        )

        # Create LLM pipeline
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=100, temperature=0.7, do_sample=True
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            retriever=retriever,
            return_source_documents=True,  # Returns retrieved docs for debugging
            verbose=True
        )

    def run(self, query: str) -> str:
        """Processes user query and returns generated response."""
        if not self.qa_chain:
            return "QA chain not initialized. Please create the QA chain first."

        response = self.qa_chain(query)
        answer = response.get("result", "No answer generated.")

        # Debug: Show retrieved docs
        retrieved_docs = response.get("source_documents", [])
        retrieved_texts = "\n\n".join([doc.page_content for doc in retrieved_docs])

        return f"Question: {query}\n\nAnswer: {answer}\n\nRetrieved Context:\n{retrieved_texts}"
