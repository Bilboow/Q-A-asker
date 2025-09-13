## 📊 Q&A Asker 🎤

This project is an **AI-powered CSV Q&A Tool** built with **Streamlit, LangChain, HuggingFace Embeddings, and ChatGroq**.  
It allows users to **upload a CSV file**, create a **vector database**, and then **ask natural language questions** about the data.  
The system retrieves the most relevant context and generates accurate answers using an LLM.


## 🚀 Features
- 📂 Upload CSV files directly from the UI
- 🧠 Create a knowledge base with Chroma + HuggingFace embeddings
- 🤖 Ask questions in natural language
- ⚡ Powered by **LangChain** + **ChatGroq**
- 🎨 Interactive Streamlit interface


## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/Q-A-asker.git
cd Q-A-asker
pip install -r requirements.txt
```

##### Your csv file should be in encoding = "cp1252" #####


## Create a .env file in the root directory with your API key:
```bash
GROQ_API_KEY_2=your_groq_api_key_here
```

## ▶️ Running the Project:
```bash
streamlit run app.py
```

## 📸 Screenshot :

<img width="1364" height="837" alt="Screenshot 2025-09-05 at 2 52 16 AM" src="https://github.com/user-attachments/assets/81d3b01d-0407-41b2-b4b8-9fe99cc34cac" />
