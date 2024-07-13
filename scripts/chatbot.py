from rasa_sdk import Action
from scripts.data_collection import get_stock_data, get_twitter_sentiment
from scripts.analysis import analyze_rsi, detect_divergence
from scripts.model import load_trained_model, predict_profitable_strategy

class ActionPredictOptionStrategy(Action):
    def name(self):
        return "action_predict_option_strategy"

    def run(self, dispatcher, tracker, domain):
        ticker = tracker.get_slot('ticker')
        stock_data = get_stock_data(ticker)
        sentiment_data = get_twitter_sentiment(ticker)
        
        rsi_status = analyze_rsi(stock_data)
        divergence = detect_divergence(stock_data)
        sentiment_score = sum(sentiment_data) / len(sentiment_data)
        
        prediction_input = {
            'RSI': rsi_status,
            'Sentiment': sentiment_score,
            'Price Change': stock_data['Close'].pct_change().iloc[-1]
        }
        model = load_trained_model()
        strategy = predict_profitable_strategy(model, prediction_input)
        
        dispatcher.utter_message(f"The recommended option strategy for {ticker} is: {strategy}")
