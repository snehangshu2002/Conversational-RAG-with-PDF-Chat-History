
# 📖 Conversational RAG with PDF + Chat History (Streamlit Chat Style)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**.  
It allows users to upload PDFs, ask questions, and get **context-aware answers** while maintaining **chat history**.  
The system uses **Groq + LangChain** for reasoning and **Google Generative AI embeddings** for semantic search.

---

## 🚀 Features
- 📂 **Upload multiple PDFs** and query them interactively.  
- 💬 **Conversational memory** with chat history per session.  
- 🔍 **Context-aware query reformulation** (understands follow-up questions).  
- 🧠 Powered by:
  - **Groq LLM (llama-3.3-70b-versatile)**  
  - **Google Gemini embeddings (gemini-embedding-001)**  
  - **FAISS Vector Store** for fast retrieval.  
- ⚡ Optimized for **Streamlit Chat UI**.  

---

## 📦 Installation

Clone the repository:
```bash
git clone https://github.com/snehangshu2002/Conversational-RAG-with-PDF-Chat-History.git
cd Conversational-RAG-with-PDF-Chat-History

````

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
LANGCHAIN_API_KEY=your_langchain_api_key
GOOGLE_API_KEY=your_google_api_key
```

⚠️ In Streamlit Cloud, store them under **`st.secrets`** instead.

---

## ▶️ Run the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 🖥️ Usage

1. Enter your **Groq API key** in the sidebar.
2. Upload one or more **PDF documents**.
3. Ask questions in natural language (e.g., *“What does section 2 say about liability?”*).
4. Continue the conversation — the bot remembers context.

---

## ⚙️ Tech Stack

* **Frontend**: Streamlit (chat-style interface)
* **LLM**: Groq (`llama-3.3-70b-versatile`)
* **Embeddings**: Google Generative AI (`gemini-embedding-001`)
* **Vector Store**: FAISS
* **Orchestration**: LangChain

---

## 📂 Project Structure

```
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
└── README.md             # Documentation
```

---

## 🛠️ Requirements

* Python 3.10+
* Streamlit
* LangChain
* FAISS
* Groq API key
* Google API key

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Streamlit](https://streamlit.io/)
* [Groq](https://groq.com/)
* [Google Generative AI](https://ai.google.dev/)

---
## 📸 Screenshot

Here’s how the app looks when running:

![App Screenshot](Screenshot-2025-08-29-221201.png)

