import os
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20B",
    api_key=os.getenv("GROQ_API_KEY_2"),
    temperature=0.1
)

vector_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
file_path = "./chroma_store"

def create_vector_db(file):
    #df = pd.read_csv("python/QA asker/codebasics_faqs.csv", encoding="cp1252")

    #docs = []
    #for _, row in df.iterrows():
     #   content = " ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
     #   docs.append(Document(page_content=content))
     loader = CSVLoader(file, encoding = "cp1252")
     docs = list(loader.load())

     vector_db = Chroma.from_documents(
        docs,
        vector_embedding,
        persist_directory=file_path
        )
    # vector_db.persist()  # optional for Chroma>=0.4
     return vector_db

def get_qa_chain():
    vector_db = Chroma(
        persist_directory=file_path,
        embedding_function=vector_embedding
    )
    retriever = vector_db.as_retriever()

    prompt_temp = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_temp,
        input_variables=['context','question']
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    response = chain({"query": "Do you have javascript course?"})
    print("Answer:", response["result"])
    
