import streamlit as st
import pandas as pd
import os
import alpaca_trade_api as tradeapi
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Set Page Config
st.set_page_config(page_title="Llama Trader AI - Options Trading Strategy Suggestor", page_icon="💹", layout="centered")
st.title("Llama Trader AI")
st.image("./images/logo.jpg")

# Load environment variables from .env file
load_dotenv()

# Access environment variables
alpaca_api_key = os.getenv("APCA_API_KEY_ID")
alpaca_secret_key = os.getenv("APCA_API_SECRET_KEY")
base_url = os.getenv("BASE_URL")
# Initialize Alpaca API
alpaca_api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, base_url=base_url, api_version="v2")

# Add custom CSS styling
st.markdown(
    """
    <style>
    h1 {
        text-shadow: 2px 2px 4px #cccccc;
        text-align: center;
    }
    .st-emotion-cache-1gv3huu, .st-emotion-cache-4oy321, .st-emotion-cache-1gulkj5, .st-emotion-cache-s1k4sy {
    background-color: rgb(240, 242, 246);
    box-shadow: 2px 2px 0px 0px #ccc;
    }
    .st-emotion-cache-1sno8jx p {
    color: #000;
}
    .user-message {
        background-color: rgb(135 181 189);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 5px;
        color: #000;
        font-weight: bold;
        box-shadow: 3px 3px 3px #ccc;
    }
    .bot-message {
        background-color: rgb(253 254 249);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 5px;
        box-shadow: 3px 3px 3px #333333;
    }
    .sidebar .user-message, .sidebar .bot-message {
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Add a welcome message
st.chat_message("assistant").write("Hi, I'm Gordon. How can I help you today?")

# StreamHandler to intercept streaming output from the LLM
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Define system prompt
system_prompt = """You are an intelligent financial assistant specialized in suggesting options trading strategies. When a user queries a stock symbol, analyze live market data and sentiment analysis. Provide the best options strategies with clear, short justifications. Use the following key metrics: RSI, MACD, Options Greeks, Implied Volatility, volume changes, and social sentiment. 
Keep responses to 250 words or less, and respond in markdown format only."""

# Set user prompt template
user_prompt_template = """You are interacting with a sophisticated financial assistant designed to help you with options trading strategies. Please provide the stock ticker you are interested in, and the assistant will analyze current market data, technical indicators, and social sentiment to recommend up to three options strategies with clear justifications.
{history}
Question: {question}
Answer:"""

# Combine system prompt and user prompt
prompt_template = system_prompt + "\n" + user_prompt_template
prompt = PromptTemplate(template=prompt_template, input_variables=["history", "question"])

stream_handler = StreamHandler(st.empty())
callback_manager = CallbackManager([stream_handler])

# Initialize the LLM & Embeddings
llm = LlamaCpp(model_path="./models/llama-2-7b-chat.Q4_K_M.gguf", callback_manager=callback_manager, temperature=1, top_k=100, top_p=.7, n_ctx=2048, n_batch=512, n_gpu_layers=1, verbose=True, f16_kv=True, streaming=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Clear Conversation Button
if st.button("Clear Conversation", key="clear_conversation"):
    memory.clear()
    st.session_state.selected_conversation = None

# Function to load documents from uploaded files
@st.cache_data
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(file_path)

        expected_columns = ['title', 'company', 'job_description', 'key_skills']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"Columns {expected_columns} not found in {file_path}")

        for _, row in df.iterrows():
            document = f"Title: {row['title']}\n"
            document += f"Company: {row['company']}\n"
            document += f"\n{row['job_description']}\n"
            document += f"Key Skills: {row['key_skills']}\n\n"
            documents.append(document)
    return documents

# Function to split documents
@st.cache_data
def split_documents(documents):
    text_splitter = SemanticChunker(embeddings(chunk_size=100, chunk_overlap=20))
    return text_splitter.split_documents(documents)

# Function to input documents into vector database
@st.cache_data
def input_into_vectordb(documents):
    return Chroma.from_documents(documents, embeddings)

# Allow .csv files to be uploaded
uploaded_files = st.file_uploader("Upload CSV Files (optional)", accept_multiple_files=True, type=["csv"])

# Process uploaded files (Optional)
if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        documents = load_documents(uploaded_files)
        st.write(f"Loaded {len(uploaded_files)} files.")
        st.write("Creating embeddings and vector database...")
        docs = split_documents(documents)
        vectorstore = input_into_vectordb(docs)
        st.write("Creating conversation chain...")
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.vectorstore = vectorstore
        st.success(f"Processed {len(uploaded_files)} files and created conversation chain.")
else:
    st.session_state.vectorstore = None

# Chat history storage with titles
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_conversation' not in st.session_state:
    st.session_state.selected_conversation = None

# Chat history sidebar with clickable titles and hover effect
with st.sidebar:
    st.subheader("Chat History")

    # Display selected conversation
    if st.session_state.selected_conversation is not None:
        selected_conversation = st.session_state.chat_history[st.session_state.selected_conversation]
        with st.expander(selected_conversation["title"]):
            for message in selected_conversation["messages"]:
                if message["role"] == "user":
                    st.markdown(f"<div class='user-message'>👨 {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot-message'>🤖 {message['content']}</div>", unsafe_allow_html=True)

# Query through LLM
question = st.chat_input("Give me a stock and I will provide some trading strategies!")
if question:
    # Check if the query should start a new conversation
    if question.startswith("What is the best options strategy"):
        st.session_state.selected_conversation = None

    context = memory.buffer

    with st.spinner("Analyzing data. Please wait..."):
        response = llm_chain.run({"history": context, "question": question})

    # Append to chat history with title
    if not st.session_state.chat_history or st.session_state.selected_conversation is None:
        st.session_state.chat_history.append({"title": question, "messages": []})
        st.session_state.selected_conversation = len(st.session_state.chat_history) - 1

    st.session_state.chat_history[st.session_state.selected_conversation]["messages"].append({"role": "user", "content": question})
    st.session_state.chat_history[st.session_state.selected_conversation]["messages"].append({"role": "assistant", "content": response})

    # Display the conversation in the main chat window
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(response)

    # Update memory
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response)

    # Automatically save chat history to a text file after each interaction
    with open("chat_history.txt", "a") as f:
        f.write(f"User: {question}\n")
        f.write(f"Bot: {response}\n\n")