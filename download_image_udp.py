import socket
import pickle
import numpy as np
import cv2
from image_detach_rebuild import redraw_image

# Configuration
HOST = 'localhost'
PORT = 10010
IMAGE_SIZE = (300, 300, 3)  # Change this to the size of your original image

def receive_pieces():
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
                print(f"Received piece at position ({x}, {y}, {c})")
                # Rebuild the image dynamically
                reconstructed_image = redraw_image(piece, reconstructed_image)
                cv2.imshow('Reconstructed Image', reconstructed_image)
                cv2.waitKey(1)
            except (Exception, UnicodeDecodeError) as e:
                print(f"An error occurred: {e}")
                continue
    cv2.destroyAllWindows()

if __name__ == "__main__":
    receive_pieces()