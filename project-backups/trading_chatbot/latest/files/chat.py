import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd
import numpy as np
import requests

# Access environment variables
alpaca_api_key = os.getenv("APCA_API_KEY_ID")
alpaca_secret_key = os.getenv("APCA_API_SECRET_KEY")
base_url = os.getenv("BASE_URL")
# Initialize Alpaca API
alpaca_api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, base_url=base_url, api_version="v2")

# Initialize Streamlit
st.set_page_config(page_title="Options Trading Assistant", layout="wide")
st.title("Options Trading Assistant")

# Sidebar for user interaction
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def fetch_sentiment(ticker):
    """Fetch sentiment from a placeholder web scraper or API."""
    sentiment_api = f"https://fakesentimentapi.com/sentiment?ticker={ticker}"
    try:
        response = requests.get(sentiment_api)
        if response.status_code == 200:
            data = response.json()
            return data.get("sentiment", "Neutral")
        else:
            return "Neutral"
    except:
        return "Neutral"

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_stock_data(ticker):
    """Fetch stock data and calculate RSI and moving averages."""
    try:
        bars = alpaca_api.get_bars(ticker, TimeFrame.Day, "2023-01-01", "2023-12-31").df
        bars.index = pd.to_datetime(bars.index)
        bars["200_MA"] = bars["close"].rolling(window=200).mean()
        bars["RSI"] = calculate_rsi(bars)
        return bars
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {str(e)}")
        return None

def analyze_ticker(ticker):
    """Analyze ticker and return recommendations."""
    data = fetch_stock_data(ticker)
    if data is not None:
        sentiment = fetch_sentiment(ticker)
        current_rsi = data["RSI"].iloc[-1]
        current_price = data["close"].iloc[-1]
        moving_avg = data["200_MA"].iloc[-1]

        st.write(f"### Analysis for {ticker}")
        st.write(f"**Current Price**: ${current_price:.2f}")
        st.write(f"**200-Day Moving Average**: ${moving_avg:.2f}")
        st.write(f"**RSI**: {current_rsi:.2f}")
        st.write(f"**Sentiment**: {sentiment}")

        if current_rsi < 30:
            st.success("RSI indicates the stock is oversold. Consider bullish strategies.")
        elif current_rsi > 70:
            st.warning("RSI indicates the stock is overbought. Consider bearish strategies.")
        else:
            st.info("RSI is neutral. Strategies should align with overall sentiment and technicals.")

        if current_price > moving_avg:
            st.success("The stock is trading above its 200-day moving average, indicating a bullish trend.")
        else:
            st.warning("The stock is trading below its 200-day moving average, indicating a bearish trend.")

# LLM Configuration
llm = OpenAI(temperature=0.7)

# Vectorstore Setup
loader = DirectoryLoader("data", glob="**/*.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

# Conversational RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

# Main interaction logic
with st.form("query_form"):
    user_input = st.text_input("Ask a question about trading strategies or tickers:", "e.g., What options strategy should I use for AAPL?")
    submitted = st.form_submit_button("Submit")

if submitted:
    if user_input.startswith("Analyze "):
        ticker = user_input.split(" ")[1].strip()
        analyze_ticker(ticker)
    else:
        response = qa_chain.run(user_input)
        st.session_state["messages"].append(("user", user_input))
        st.session_state["messages"].append(("bot", response))

# Display conversation history
for role, msg in st.session_state["messages"]:
    if role == "user":
        st.markdown(f"**You**: {msg}")
    else:
        st.markdown(f"**Bot**: {msg}")
