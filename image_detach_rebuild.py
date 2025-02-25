import numpy as np
from sklearn.utils import shuffle


def detach_image(image, piece_size=(30, 30)):
    """
    Detach an RGB image into small parts and return them in a shuffled list.

    Parameters:
    image (np.ndarray): The input RGB image to be detached.
    piece_size (tuple): The size of each piece (height, width).

    Returns:
    list: A shuffled list of tuples where each tuple contains the piece and its original position.
    """
    height, width, channels = image.shape
    assert channels == 3, "Image must have 3 channels (RGB)."

    pieces = []
    for y in range(0, height, piece_size[1]):
        for x in range(0, width, piece_size[0]):
            for c in range(channels):
                piece = image[y:y + piece_size[1], x:x + piece_size[0], c:c + 1]
                pieces.append(((y, x, c), piece))

    shuffled_pieces = shuffle(pieces)
    return shuffled_pieces

def rebuild_image(pieces, image_size, piece_size=(30, 30)):
    """
    Rebuild the image from the detached pieces.

    Parameters:
    pieces (list): A list of tuples where each tuple contains the piece and its original position.
    image_size (tuple): The size of the original image (height, width, channels).
    piece_size (tuple): The size of each piece (height, width).

    Returns:
    np.ndarray: The reconstructed image.
    """
    # height, width, channels = image_size
    reconstructed_image = np.zeros(image_size, dtype=np.uint8)

    for (y, x, c), piece in pieces:
        reconstructed_image[y:y + piece_size[1], x:x + piece_size[0], c:c + 1] = piece

    return reconstructed_image

def redraw_image(patch, reconstructed_image, piece_size=(30, 30)):
    (y, x, c), piece = patch

    reconstructed_image[y:y + piece_size[1], x:x + piece_size[0], c:c + 1] = piece

    return reconstructed_image
