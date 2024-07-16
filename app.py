import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from collections import Counter

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://jeff:webmaster@localhost:3306/aibot'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

API_KEY = '332e851efa9c4549886ca84b1ffb8196'
BASE_URL = 'https://newsapi.org/v2/everything'

CATEGORIES = ['technology', 'sports', 'politics', 'entertainment', 'business']


class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class SimpleClassifier:
    def __init__(self):
        self.category_keywords = {
            'technology': ['tech', 'software', 'hardware', 'AI', 'robot', 'computer', 'internet', 'app', 'digital',
                           'cyber'],
            'sports': ['sport', 'football', 'basketball', 'tennis', 'game', 'player', 'team', 'athlete', 'league',
                       'tournament'],
            'politics': ['politic', 'government', 'election', 'party', 'president', 'vote', 'law', 'policy', 'congress',
                         'senate', 'protests'],
            'entertainment': ['movie', 'music', 'celebrity', 'film', 'star', 'TV', 'show', 'actor', 'singer',
                              'Hollywood'],
            'business': ['business', 'economy', 'market', 'stock', 'company', 'trade', 'finance', 'investment',
                         'startup', 'entrepreneur']
        }

    def classify(self, text):
        text = text.lower()
        scores = {category: 0 for category in self.category_keywords}
        words = text.split()

        for word in words:
            for category, keywords in self.category_keywords.items():
                if any(keyword in word for keyword in keywords):
                    scores[category] += 1

        # If no category scores any points, return 'general'
        if all(score == 0 for score in scores.values()):
            return 'general'

        return max(scores, key=scores.get)

    def update_from_history(self):
        queries = Query.query.all()
        for query in queries:
            words = query.user_input.lower().split()
            self.category_keywords[query.category].extend(words)

        for category in self.category_keywords:
            self.category_keywords[category] = list(set(self.category_keywords[category]))


classifier = SimpleClassifier()

MAX_WORDS = 1000
MAX_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
label_encoder = LabelEncoder()


class TFClassifier:
    def __init__(self):
        self.model = self.create_model()
        self.trained = False

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(MAX_WORDS, 16, input_length=MAX_LENGTH),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(len(CATEGORIES), activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def prepare_text(self, text):
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
        return padded

    def train(self, texts, labels):
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

        encoded_labels = label_encoder.fit_transform(labels)
        categorical_labels = tf.keras.utils.to_categorical(encoded_labels)

        self.model.fit(padded_sequences, categorical_labels, epochs=10, verbose=1)
        self.trained = True

    def classify(self, text):
        if not self.trained:
            return 'general'  # Default category if not trained
        padded_sequence = self.prepare_text(text)
        prediction = self.model.predict(padded_sequence)
        category_index = np.argmax(prediction[0])
        return CATEGORIES[category_index]

    def update_from_history(self):
        queries = Query.query.all()
        texts = [query.user_input for query in queries]
        labels = [query.category for query in queries]
        self.train(texts, labels)


classifier2 = TFClassifier()


def fetch_news(topic, days=1):
    today = datetime.now().date()
    from_date = (today - timedelta(days=days)).isoformat()

    params = {
        'q': topic,
        'from': from_date,
        'sortBy': 'publishedAt',
        'apiKey': API_KEY,
        'language': 'en'
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        return None


def format_news(articles, limit=5):
    """
    formatted = []
    for article in articles[:limit]:
        formatted.append(f"Title: {article['title']}\nDescription: {article['description']}\nURL: {article['url']}\n")
    return '\n'.join(formatted)
    """

    formatted = []
    for article in articles[:limit]:
        formatted.append({
            'title': article['title'],
            'description': article['description'],
            'url': article['url']
        })
    return formatted

@app.route("/")
def index():
    # return "<p>Hello, World!</p>"
    return render_template('index.html')


@app.route('/get_news', methods=['POST'])
def get_news():
    user_input = request.form['user_input']
    category = classifier.classify(user_input)

    # Store query in database
    new_query = Query(user_input=user_input, category=category)
    db.session.add(new_query)
    db.session.commit()

    search_term = user_input if category == 'general' else f"{user_input} {category}"
    articles = fetch_news(search_term)
    if articles:
        formatted_articles = format_news(articles)
        first_article_category = classifier.classify(articles[0]['title'] + ' ' + articles[0]['description'])
        return jsonify({
            'success': True,
            'category': category,
            'articles': formatted_articles,
            'first_article_category': first_article_category
        })
    else:
        return jsonify({
            'success': False,
            'message': f"Sorry, I couldn't find any recent news about '{user_input}'."
        })


if __name__ == "__main__":
    # chatbot()
    with app.app_context():
        db.create_all()
        classifier.update_from_history()
    app.run(debug=True)
