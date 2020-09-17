from gevent import monkey
monkey.patch_all()
from flask import Flask, render_template
from flask_socketio import SocketIO, emit, disconnect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
# socketio = SocketIO(app, cors_allowed_origins="*", message_queue='redis://', async_mode='eventlet')
socketio = SocketIO(app, async_mode="threading") # async_mode=None, logger=True, engineio_logger=True)

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
    emit('my_response', {'data': message})
    # socketio.sleep(2)
    socketio.disconnect()

def on_disconnect(self):
    print('Client disconnected')
        

if __name__ == '__main__':
    socketio.run(app)