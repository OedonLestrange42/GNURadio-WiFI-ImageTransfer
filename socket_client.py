import socket
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_detach_rebuild import rebuild_image

# Configuration
HOST = 'localhost'
PORT = 65432
IMAGE_SIZE = (300, 300, 3)  # Change this to the size of your original image

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
            img_bgr = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Reconstructed Image', img_bgr)
            cv2.waitKey(1)
    """plt.imshow(reconstructed_image)
    plt.axis('off')  # Hide the axis
    plt.show()"""
    # Hold the window open until a key is pressed
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    receive_pieces()