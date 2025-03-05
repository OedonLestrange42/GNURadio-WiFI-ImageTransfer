import pdb

import numpy as np
import socket
import pickle
import struct
import time
import torch
import os
import threading
from codec.jsce_codec import JSCE
from PIL import Image
from image_detach_rebuild import detach_image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HOST = 'localhost'
IMAGE_SIZE = (240, 240, 3)
TARGET = ['3-4', '13-10']
stop_flag = threading.Event()


def send_image(image_path1, image_path2, port):
    global stop_flag
    original_image1 = Image.open(image_path1).convert('RGB')
    original_image2 = Image.open(image_path2).convert('RGB')
    image_dict = {TARGET[0]: original_image1, TARGET[0]: original_image2}

    img_latent = codec.img2msg(image_dict)
    pieces = detach_image(img_latent)
    data = pickle.dumps(pieces[0])
    print('Image to Pieces Accomplished. Each piece size:', len(data))
    # pdb.set_trace()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while not stop_flag.is_set():
            for piece in pieces:
                if stop_flag.is_set():
                    break
                data = pickle.dumps(piece)
                # message_size = struct.pack("=L", len(data))
                # print(len(data))
                s.sendto(data, (HOST, port))
                time.sleep(0.05)
            print("loop accomplished")

@app.route('/', methods=['GET'])
def index():
    return render_template('featuremap_sender.html')

@app.route('/send_image', methods=['POST'])
def handle_send_image():
    global stop_flag
    file1 = request.files['file1']
    file2 = request.files['file2']
    port = request.form['port']
    if file1 and file2 and port:
        file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
        file1.save(file_path1)
        file2.save(file_path2)

        stop_flag.clear()
        threading.Thread(target=send_image, args=(file_path1, file_path2, int(port))).start()
        return jsonify({"status": "sending"})
    return jsonify({"status": "error"})

@app.route('/stop', methods=['POST'])
def handle_stop():
    global stop_flag
    stop_flag.set()
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    codec = JSCE(weight_path='codec/checkpoints/Rician-checkpoint_SOMA-DSCN_withIRS_optIRS_IRS-scale-8_AP-1_Usr-5_img-size-64_epoch400_20241216.pth',
                 img_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                 compressed_channel=128,
                 device=device)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5001)