from flask import Flask, render_template
from flask_socketio import SocketIO, emit, disconnect
from threading import Lock
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, engineio_logger=True, async_mode=async_mode)
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
         {'data': message})

def on_disconnect(self):
    print('Client disconnected')
        

if __name__ == '__main__':
    socketio.run(app, debug=True, threaded=True)