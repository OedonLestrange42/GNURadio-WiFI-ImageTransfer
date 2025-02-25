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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")  # Add cors_allowed_origins

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
                (x, y, c), val = piece

                # Rebuild the image dynamically
                reconstructed_image = redraw_image(piece, reconstructed_image)
                
                # Convert to BGR before encoding (if the image is in RGB)
                img_bgr = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)
                
                # Ensure the image is properly encoded with high quality
                _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                print(f"Emitting image update (data length: {len(jpg_as_text)})")
                socketio.emit('update_image', {'image': jpg_as_text}, namespace='/')
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

@app.route('/')
def index():
    return render_template('receiver.html')

def start_receiver():
    receive_pieces(socketio)

if __name__ == "__main__":
    threading.Thread(target=start_receiver).start()
    socketio.run(app, debug=True)