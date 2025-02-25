import socket
import pickle
import numpy as np
import cv2
from image_detach_rebuild import rebuild_image

# Configuration
HOST = 'localhost'
PORT = 65432
IMAGE_SIZE = (256, 256, 3)  # Change this to the size of your original image

def receive_pieces():
    pieces = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            data = s.recv(4096)
            if not data:
                break
            piece = pickle.loads(data)
            pieces.append(piece)
            s.sendall(b'ACK')
            # Rebuild the image dynamically
            reconstructed_image = rebuild_image(pieces, IMAGE_SIZE)
            cv2.imshow('Reconstructed Image', reconstructed_image)
            cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    receive_pieces()