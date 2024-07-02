import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

API_KEY = '332e851efa9c4549886ca84b1ffb8196'
BASE_URL = 'https://newsapi.org/v2/everything'

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
    app.run(debug=True)
