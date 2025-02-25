from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from image_detach_rebuild import detach_image
import socket
import pickle
import struct
import time
import os
import threading

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HOST = 'localhost'
stop_flag = threading.Event()

def send_image(image_path, port):
    global stop_flag
    original_image = np.array(Image.open(image_path).resize((300, 300)), dtype=np.uint8)
    pieces = detach_image(original_image)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while not stop_flag.is_set():
            for piece in pieces:
                data = pickle.dumps(piece)
                message_size = struct.pack("=L", len(data))
                # print(len(data))
                s.sendto(data, (HOST, port))
                time.sleep(0.03)
            print("loop accomplished")

@app.route('/', methods=['GET'])
def index():
    return render_template('sender.html')

@app.route('/send_image', methods=['POST'])
def handle_send_image():
    global stop_flag
    file = request.files['file']
    port = request.form['port']
    if file and port:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        stop_flag.clear()
        threading.Thread(target=send_image, args=(file_path, int(port))).start()
        return jsonify({"status": "sending"})
    return jsonify({"status": "error"})

@app.route('/stop', methods=['POST'])
def handle_stop():
    global stop_flag
    stop_flag.set()
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5001)