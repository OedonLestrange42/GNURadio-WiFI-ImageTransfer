import socket
import pickle
import numpy as np
from PIL import Image
from image_detach_rebuild import detach_image

# Configuration
HOST = 'localhost'
PORT = 65432
IMAGE_PATH = 'images/kodim03.png'

# Load image and detach it into pieces
# image = np.random.randint(0, 256, (90, 90, 3), dtype=np.uint8)  # Replace this with image loading
image = np.array(Image.open(IMAGE_PATH).resize((300, 300)), dtype=np.uint8)
pieces = detach_image(image)

def serve_pieces():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f'Server listening on {HOST}:{PORT}')
        conn, addr = s.accept()
        with conn:
            print(f'Connected by {addr}')
            for piece in pieces:
                data = pickle.dumps(piece)
                conn.sendall(data)
                ack = conn.recv(1024)
                if ack != b'ACK':
                    break
            print('All pieces sent.')

if __name__ == "__main__":
    serve_pieces()