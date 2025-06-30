import eventlet

eventlet.monkey_patch()  # Ensure this is called before any other imports

import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


def generate_random_image():
    while True:
        # Generate a random NumPy array
        random_array = np.random.randint(0, 256, (100, 100, 3), dtype='uint8')

        # Convert the NumPy array to an image
        img = Image.fromarray(random_array, 'RGB')

        # Save the image to a buffer
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        # Encode the image as base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Emit the image to the client
        socketio.emit('image_update', {'image': f'data:image/jpeg;base64,{img_base64}'})

        # Sleep for 5 seconds before generating the next image
        eventlet.sleep(0.1)


@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(generate_random_image)


if __name__ == '__main__':
    socketio.run(app, debug=True, host='localhost', port=5002)