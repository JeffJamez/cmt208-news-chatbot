import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from collections import Counter
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
            'technology': ['tech', 'software', 'hardware', 'AI', 'robot', 'computer', 'internet'],
            'sports': ['sport', 'football', 'basketball', 'tennis', 'game', 'player', 'team'],
            'politics': ['politic', 'government', 'election', 'party', 'president', 'vote'],
            'entertainment': ['movie', 'music', 'celebrity', 'film', 'star', 'TV', 'show'],
            'business': ['business', 'economy', 'market', 'stock', 'company', 'trade', 'finance']
        }

    def classify(self, text):
        text = text.lower()
        scores = {category: sum(keyword in text for keyword in keywords)
                  for category, keywords in self.category_keywords.items()}
        return max(scores, key=scores.get)

    def update_from_history(self):
        queries = Query.query.all()
        for query in queries:
            words = query.user_input.lower().split()
            self.category_keywords[query.category].extend(words)

        for category in self.category_keywords:
            self.category_keywords[category] = list(set(self.category_keywords[category]))


classifier = SimpleClassifier()


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


'''
def chatbot():
    print("Welcome to the News Chatbot!")
    print("You can ask for news on any topic. Type 'quit' to exit.")

    while True:
        user_input = input("What news would you like to read about? ")

        if user_input.lower() == 'quit':
            print("Thank you for using News Chatbot. Goodbye!")
            break

        articles = fetch_news(user_input)
        if articles:
            print(f"Here are the latest news articles about '{user_input}':")
            print("\n")
            print(format_news(articles))
        else:
            print(f"Sorry, I couldn't find any recent news about '{user_input}'.")
'''


@app.route("/")
def index():
    # return "<p>Hello, World!</p>"
    return render_template('index.html')


@app.route('/get_news', methods=['POST'])
def get_news():
    user_input = request.form['user_input']
    category = classifier.classify(user_input)

    new_query = Query(user_input=user_input, category=category)
    db.session.add(new_query)
    db.session.commit()

    articles = fetch_news(user_input)
    if articles:
        formatted_articles = format_news(articles)
        return jsonify({
            'success': True,
            'articles': formatted_articles
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
