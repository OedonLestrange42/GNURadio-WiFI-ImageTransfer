from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np
from sklearn.utils import shuffle
import socket
import struct
import pickle
import time
from PIL import Image
from image_detach_rebuild import detach_image

HOST = 'localhost'


def send_image(port):

    original_image = np.array(Image.open('target.jpg').resize((300, 300)), dtype=np.uint8)
    pieces = detach_image(original_image)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while True:
            for piece in pieces:
                data = pickle.dumps(piece)
                message_size = struct.pack("=L", len(data))

                s.sendto(message_size + data, (HOST, port))
                """ack = s.recv(1024)
                if ack != b'ACK':
                    break"""
                time.sleep(0.003)
            print("loop accomplished")

            # print('All pieces sent.')

    # socketio.emit('terminal_output', {'data': 'Task stopped.'})


if __name__ == '__main__':
    # socketio.run(app, debug=True)
    send_image(50010)