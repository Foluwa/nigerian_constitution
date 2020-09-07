from flask import Flask, render_template
from flask_socketio import SocketIO, emit, disconnect
from threading import Lock

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

@app.route('/')
def hello():
    return ('Conversation AI')

@app.route('/chat')
def chat():
    return render_template('chat.html')

def on_connect(self):
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Client Connected'})

@socketio.on('my_event', namespace='/chat')
def test_message(message):
    print('message is >> ', message)
    emit('my_response',
         {'data': 'data is received!!'})

def on_disconnect(self):
        print('Client disconnected')
        

if __name__ == '__main__':
    socketio.run(app, debug=True)