#!/usr/bin/env python
from gevent import monkey
monkey.patch_all()
from flask import Flask, render_template
from flask_socketio import SocketIO, emit, disconnect
from torch_utils import get_prediction

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app,  engineio_logger=True, async_handlers=True) # async_mode=None, logger=True,)

@app.route('/')
def index():
    return ('Conversation AI')

@app.route('/chat')
def chat():
    return render_template('chat.html')

def on_connect(self):
    emit('my_response', {'data': 'Client Connected'})

@socketio.on('my_event', namespace='/chat')
def user_message(message):
    print('message is >> ', message)
    
    # load image
    
    # prediction
    prediction = get_prediction(message)
    
    emit('my_response', {'data': prediction })


def on_disconnect(self):
    print('Client disconnected')
        

if __name__ == '__main__':
    socketio.run(app, debug=True)