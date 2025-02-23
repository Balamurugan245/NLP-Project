from flask import Flask, render_template, request
from vectorizer import SpacyVectorizer  
import pickle
import os
import numpy as np

app = Flask(__name__)

vectorizer = SpacyVectorizer()

# Paths to saved models
model_paths = {
    "spam": "models/Spam_dediction.pickle",
    "movie_review": "models/movie_review.pickle",
    "emotion": "models/emotion_analysis.pickle",
    "fake_news": "models/fake_news.pickle"
}

# Label mappings for prediction results
fake_or_real_dict = {
    0: "Fake News",
    1: "Real News"
}

emotions_dict = {
    2: "joy",        
    4: "sad",        
    0: "anger",       
    1: "fear",        
    3: "love"     
}

movies_dict = {
    0: "negative review",
    1: "Positive review"
}

mail_dict = {
    0: "Not Spam",
    1: "Spam"
}

# Load models
models = {} 
for model_name, path in model_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        models[model_name] = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

def predict_text(model_key, text):
    """
    Predict the output for a given text using the specified model.
    
    Args:
        model_key (str): The key to identify the model (e.g., 'emotion').
        text (str): The input text for prediction.
    
    Returns:
        str: The predicted label from the model.
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Ensure text is not empty
    if not text.strip():
        return "Error: Empty input text"

    model = models[model_key]

    # If the model is a Pipeline, pass raw text (model will handle vectorization)
    if hasattr(model, "predict"):
        result = model.predict([text])[0]  # No need to vectorize manually
    else:
        # If the model isn't a pipeline, vectorize the text manually
        vectorized_text = vectorizer.transform([text]).reshape(1, -1)
        result = model.predict(vectorized_text)[0]

    return result


@app.route('/spam', methods=['GET', 'POST'])
def spam():
    mail_result = None
    if request.method == 'POST':
        text = request.form['text']
        result = predict_text("spam", text)
        mail_result = mail_dict.get(result, "Unknown")
    return render_template('spam.html', result=mail_result)

@app.route('/movie_review', methods=['GET', 'POST'])
def movie_review():
    movie_result = None
    if request.method == 'POST':
        text = request.form['text']
        result = predict_text("movie_review", text)
        movie_result = movies_dict.get(result, "Unknown")
    return render_template('movie_review.html', result=movie_result)

@app.route('/emotion', methods=['GET', 'POST'])
def emotion():
    emotion_result = None
    if request.method == 'POST':
        text = request.form['text']
        result = predict_text("emotion", text)
        emotion_result = emotions_dict.get(result, "Unknown")
    return render_template('emotion.html', result=emotion_result)

@app.route('/fake_news', methods=['GET', 'POST'])
def fake_news():
    news_result = None
    if request.method == 'POST':
        text = request.form['text']
        result = predict_text("fake_news", text)
        news_result = fake_or_real_dict.get(result, "Unknown")
    return render_template('fake_news.html', result=news_result)

if __name__ == '__main__':
    app.run(debug=True)
