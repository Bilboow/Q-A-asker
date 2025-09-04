import streamlit as st
import pandas as pd
from helper import get_qa_chain,create_vector_db

st.header("Q&A asker 🎤 :")

file_csv = st.file_uploader("Enter your CSV file",type=['csv'])


if file_csv is not None:
    # Save uploaded file to disk so helper.py can use it
    file_path = "uploaded_file.csv"
    with open(file_path, "wb") as f:
        f.write(file_csv.getbuffer())

    st.success("✅ File uploaded successfully!")


     # sending to vectordb
if st.button("Create Knowledge"):
    create_vector_db(file_path)
    st.success("Vector database successfully connected ✅ ")

query = st.text_input("Question :")

if query:
    chain = get_qa_chain()
    response = chain(query)

    st.header("Answer")
    st.write(response["result"])
