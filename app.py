from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hii Hello, World!'


@app.route('/chat')
def chat():
    return render_template('chat.html')