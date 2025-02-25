import socket
import struct
import pickle
import time
from PIL import Image

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # ipv4
# Assign an address and a port
host = socket.gethostname()
s_address = (host, 52002)
s.connect(s_address)

frame = Image.open('RX.jpg')
image_file = frame.convert('L')
threshold = 127
image_file = image_file.point( lambda p: 255 if p > threshold else 0 )
image_file = image_file.convert('1')

# Serialize frame
data = pickle.dumps(image_file)
# data = 'hello world!'.encode()
while True:
    print("Connected")

    # Send message length first
    message_size = struct.pack("=L", len(data))

    # Then data
    s.sendall(message_size + data)
    time.sleep(0.005)  # 5ms
s.close()
