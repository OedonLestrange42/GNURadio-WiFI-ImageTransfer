import pdb
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import socket
import pickle
import numpy as np
import cv2
import base64
from image_detach_rebuild import redraw_image

# Configuration
HOST = 'localhost'
PORT = 10011
IMAGE_SIZE = (300, 300, 3)  # Change this to the size of your original image

def receive_pieces(socketio):
    reconstructed_image = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))
        while True:
            try:
                data, client_address = s.recvfrom(4096)
                if not data:
                    print("No data received, breaking the connection loop")
                    continue
                piece = pickle.loads(data)
                # pdb.set_trace()
                (x, y, c), val = piece
                # print(f"Received piece at position ({x}, {y}, {c})")

                # Rebuild the image dynamically
                reconstructed_image = redraw_image(piece, reconstructed_image)
                img_bgr = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', img_bgr)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('update_image', {'image': jpg_as_text})
            except (Exception, UnicodeDecodeError) as e:
                print(f"An error occurred: {e}")
                continue

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('receiver.html')

def start_receiver():
    receive_pieces(socketio)

if __name__ == "__main__":
    threading.Thread(target=start_receiver).start()
    socketio.run(app, debug=True)