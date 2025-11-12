import os
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize the LLM (Groq)
llm = ChatGroq(
    model="openai/gpt-oss-20B",
    api_key=os.getenv("GROQ_API_KEY_2"),
    temperature=0.1
)

# Initialize embedding model
vector_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
file_path = "/tmp/chroma_store"

def create_vector_db(file_path_csv):
    """Creates or updates a Chroma vector store from a CSV file."""
    loader = CSVLoader(file_path_csv, encoding="cp1252")
    docs = list(loader.load())

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=vector_embedding,
        persist_directory=file_path
    )
    vector_db.persist()
    return vector_db

def get_qa_chain():
    """Loads the Chroma vector store and returns a RetrievalQA chain."""
    vector_db = Chroma(
        persist_directory=file_path,
        embedding_function=vector_embedding
    )
    retriever = vector_db.as_retriever()

    prompt_temp = """Given the following context and a question, generate an answer based on this context only.
In the answer, try to provide as much text as possible from the 'response' section in the source document without major changes.
If the answer is not found in the context, say "I don't know."

CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_temp,
        input_variables=['context', 'question']
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

if __name__ == "__main__":
    csv_path = "python/QA_asker/codebasics_faqs.csv"  # ✅ update path to your file
    create_vector_db(csv_path)
    chain = get_qa_chain()
    response = chain.invoke({"query": "Do you have a JavaScript course?"})
    print("Answer:", response["result"])
