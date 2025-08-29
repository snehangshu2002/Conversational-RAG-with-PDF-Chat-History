## Conversational RAG With PDF + Chat History (Streamlit Chat Style)
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os, asyncio, sys
from dotenv import load_dotenv

# --- Fix for Streamlit + gRPC async client issue ---
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load env vars
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Conversational RAG With PDF uploads and chat history"

# ------------------- UI -------------------
st.set_page_config(page_title="RAG Q&A Chatbot", page_icon="📚", layout="wide")

st.title("📖 Conversational RAG with PDF & Chat History")
st.caption("Upload PDFs, ask questions, and get context-aware answers powered by **Groq + LangChain**.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Enter your **Groq API key**", type="password")
    session_id = st.text_input("Session ID", value="default_session")
    uploaded_files = st.file_uploader("📂 Upload PDF(s)", type="pdf", accept_multiple_files=True)

# ------------------- Core -------------------
if api_key:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    if "store" not in st.session_state:
        st.session_state.store = {}

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Create vector DB
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "You are an assistant that reformulates user queries. "
            "You are given a chat history and the latest user question, "
            "which may refer to earlier parts of the conversation. "
            "Your task is to rewrite the latest question so it is a complete, "
            "standalone query that can be understood without seeing the chat history. "
            "Do not answer the question. Only return the reformulated question. "
            "If the question is already clear and complete, return it exactly as it is."
        )
        # Contextual retriever
        contextualize_q_prompt = ChatPromptTemplate(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # QA chain
        qa_prompt = ChatPromptTemplate(
            [
                ("system",
                 "You are a reliable assistant for Q&A. "
                 "Use ONLY the context below to answer. "
                 "If answer not in context, reply with 'I don't know'. "
                 "Keep it short (max 3 sentences) if user want detailed explanation then give this explanation too and if you know that then add some explanation if required .\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Initialize messages for Streamlit UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask something about your PDFs..."):
            # User message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Bot response
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}}
            )
            answer = response["answer"]

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

else:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to continue.")

