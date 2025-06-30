import pdb

import eventlet
eventlet.monkey_patch()
import torch
import socket
import pickle
import numpy as np
from image_detach_rebuild import redraw_image
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from PIL import Image
from codec.jsce_codec import JSCE
import base64
import io

# Configuration
HOST = 'localhost'
PORT = 10010
IMAGE_SIZE = (240, 240, 3)
FEATURE_SIZE = (10, 10, 1)
USER_ID = '3-4'  # default value

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
stop_thread = False
feature_map = np.zeros((IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8, 128), dtype=np.float32)

def receive_pieces():
    global stop_thread
    global feature_map
    print("Starting receive_pieces function...")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        print(f"Binding to {HOST}:{PORT}")
        s.bind((HOST, PORT))
        s.settimeout(1.0)  # Add timeout to allow checking stop_thread
        piece_count = 0
        
        while not stop_thread:
            try:
                data, client_address = s.recvfrom(4096)
                if not data:
                    continue
                # print(len(data))
                piece = pickle.loads(data)
                # print(piece)
                # pdb.set_trace()
                (x, y, c), val = piece
                piece_shape = val.shape
                print(f"Received piece at position ({x}, {y}, {c}), shape: {piece_shape}")

                if piece_shape == FEATURE_SIZE:
                    # print("Received one pieces")
                    feature_map = redraw_image(piece, feature_map)
                    piece_count += 1
                
                # every N steps
                if piece_count > 0 and piece_count % 3 == 0:
                    print(f"Received {piece_count} pieces")
                    piece_count = 0
                    # pdb.set_trace()
                    reconstructed_image = codec.msg2img(feature_map, USER_ID)
                    # img = Image.fromarray(reconstructed_image.astype('uint8'), 'RGB')
                    buf = io.BytesIO()
                    reconstructed_image.save(buf, format='JPEG')
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
    return render_template('featuremap_receiver.html', user_id=USER_ID)

@socketio.on('connect')
def handle_connect():
    print("Client connected")  # Debug log

@socketio.on('start_receiving')
def handle_start(data):
    global stop_thread, USER_ID
    print("Received start signal")  # Debug log
    USER_ID = data.get('user_id', '3-4')  # Update USER_ID with the value from frontend
    print(f"Using USER_ID: {USER_ID}")
    stop_thread = False
    socketio.start_background_task(receive_pieces)
    """thread = threading.Thread(target=receive_pieces)
    thread.daemon = True
    thread.start()"""

@socketio.on('stop_receiving')
def handle_stop():
    global stop_thread
    global feature_map
    # feature_map = np.zeros((IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8, 128), dtype=np.float32)
    print("Received stop signal")  # Debug log
    stop_thread = True

if __name__ == "__main__":
    # img_size = (128, 128)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    codec = JSCE(weight_path='codec/checkpoints/Rician-checkpoint_SOMA-DSCN-exp-ver_noIRS_fixIRS_AP-1_Usr-5_img-size-128_epoch100_20250306.pth',
                 img_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                 compressed_channel=128,
                 device=device)
    socketio.run(app, host=HOST, port=5000, debug=True, allow_unsafe_werkzeug=True)