import requests
import tweepy
import pandas as pd

def get_stock_data(ticker):
    api_key = '76ZLGR0W1WWFRIO2'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data['Time Series (Daily)']).transpose()

def get_twitter_sentiment(ticker):
    consumer_key = 'qjImNLGElxOiuIqDkQgZH4cCu'
    consumer_secret = 'wuM7UhH1Ujz6EAjaR7IRcrJV6HP9zfP9LaO3STgACy23WJ8N4C'
    access_token = '128639496-FmRJc8wXp7aozJyIYpzjRc3yCblEpC1Uj0WBnvHu'
    access_token_secret = '1yBXTOMYrR8NNvhG9SKPMvMlSNCgTAwyj77p6ghOsPb4a'
    
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    
    tweets = api.search(q=ticker, lang='en', count=100)
    sentiments = [analyze_sentiment(tweet.text) for tweet in tweets]
    return sentiments

def analyze_sentiment(text):
    sentiment_score = simple_sentiment_analysis(text)
    return sentiment_score

def simple_sentiment_analysis(text):
    # Placeholder for a real sentiment analysis function
    return 1 if 'good' in text else -1 if 'bad' in text else 0
