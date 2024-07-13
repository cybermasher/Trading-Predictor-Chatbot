from sklearn.ensemble import RandomForestClassifier

def train_model(data):
    features = data[['RSI', 'Sentiment', 'Price Change']]
    target = data['Option Profit']
    model = RandomForestClassifier()
    model.fit(features, target)
    return model

def predict_profitable_strategy(model, new_data):
    prediction = model.predict(new_data)
    return prediction

def load_trained_model():
    # Placeholder for model loading logic
    return RandomForestClassifier()  # Replace with actual model loading
