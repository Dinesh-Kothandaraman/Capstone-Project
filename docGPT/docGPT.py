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
