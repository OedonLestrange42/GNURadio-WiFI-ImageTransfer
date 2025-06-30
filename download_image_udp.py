import eventlet
eventlet.monkey_patch()

import socket
import pickle
import numpy as np
from image_detach_rebuild import redraw_image
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from PIL import Image
import base64
import io

# Configuration
HOST = 'localhost'
PORT = 10010
IMAGE_SIZE = (300, 300, 3)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
stop_thread = False
reconstructed_image = np.zeros(IMAGE_SIZE, dtype=np.uint8)

def receive_pieces():
    global stop_thread
    global reconstructed_image
    print("Starting receive_pieces function...")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        print(f"Binding to {HOST}:{PORT}")
        s.bind((HOST, PORT))
        s.settimeout(1.0)  # Add timeout to allow checking stop_thread
        
        while not stop_thread:
            try:
                data, client_address = s.recvfrom(2048)
                if not data:
                    continue
                    
                piece = pickle.loads(data)
                (x, y, c), val = piece
                print(f"Received piece at position ({x}, {y}, {c})")
                
                reconstructed_image = redraw_image(piece, reconstructed_image)
                
                # Ensure proper image conversion and encoding
                # img_bgr = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)
                # _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img = Image.fromarray(reconstructed_image.astype('uint8'), 'RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                socketio.emit('image_update', {'image': f'data:image/jpeg;base64,{img_base64}'})
                eventlet.sleep(0.02)
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

@app.route('/')
def index():
    return render_template('receiver.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")  # Debug log

@socketio.on('start_receiving')
def handle_start():
    global stop_thread
    print("Received start signal")  # Debug log
    stop_thread = False
    socketio.start_background_task(receive_pieces)
    """thread = threading.Thread(target=receive_pieces)
    thread.daemon = True
    thread.start()"""

@socketio.on('stop_receiving')
def handle_stop():
    global stop_thread
    print("Received stop signal")  # Debug log
    stop_thread = True

if __name__ == "__main__":
    socketio.run(app, host=HOST, port=5000, debug=True, allow_unsafe_werkzeug=True)