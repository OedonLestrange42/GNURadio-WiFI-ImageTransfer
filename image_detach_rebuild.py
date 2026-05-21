import numpy as np
from sklearn.utils import shuffle

PIECE_SIZE = (10, 10)
# Larger spatial patches for full-RGB chunking (e.g. neural bypass in download_image_udp.py).
PATCH_PIECE_SIZE = (50, 50)

def _normalize_piece_size(piece_size):
    piece_h, piece_w = piece_size
    if piece_h <= 0 or piece_w <= 0:
        raise ValueError("piece_size values must be positive")
    return int(piece_h), int(piece_w)

def detach_image(image, piece_size=PIECE_SIZE):
    """
    Detach an RGB image into small parts and return them in a shuffled list.

    Parameters:
    image (np.ndarray): The input RGB image to be detached.
    piece_size (tuple): The size of each piece (height, width).

    Returns:
    list: A shuffled list of tuples where each tuple contains the piece and its original position.
    """
    height, width, channels = image.shape
    piece_h, piece_w = _normalize_piece_size(piece_size)
    # assert height < piece_size[0] and width < piece_size[1], "Image is too small."
    # assert channels == 3, "Image must have 3 channels (RGB)."
    piece_h = min(piece_h, height)
    piece_w = min(piece_w, width)

    pieces = []
    for y in range(0, height, piece_h):
        for x in range(0, width, piece_w):
            for c in range(channels):
                piece = image[y:y + piece_h, x:x + piece_w, c:c + 1]
                pieces.append(((y, x, c), piece))

    shuffled_pieces = shuffle(pieces)
    return shuffled_pieces


def detach_image_patches(image, piece_size=PATCH_PIECE_SIZE):
    """
    Detach an RGB image into spatial patches (all channels together) and return them shuffled.

    Each piece is ((y, x, 0), patch_rgb) where patch_rgb has shape (h, w, 3).
    """
    height, width, _channels = image.shape
    piece_h, piece_w = _normalize_piece_size(piece_size)
    piece_h = min(piece_h, height)
    piece_w = min(piece_w, width)

    pieces = []
    for y in range(0, height, piece_h):
        for x in range(0, width, piece_w):
            piece = image[y:y + piece_h, x:x + piece_w, :]
            pieces.append(((y, x, 0), piece))

    return shuffle(pieces)


def rebuild_image(pieces, image_size, piece_size=PIECE_SIZE):
    """
    Rebuild the image from the detached pieces.

    Parameters:
    pieces (list): A list of tuples where each tuple contains the piece and its original position.
    image_size (tuple): The size of the original image (height, width, channels).
    piece_size (tuple): The size of each piece (height, width).

    Returns:
    np.ndarray: The reconstructed image.
    """
    height, width, channels = image_size
    piece_h, piece_w = _normalize_piece_size(piece_size)
    piece_h = min(piece_h, height)
    piece_w = min(piece_w, width)

    reconstructed_image = np.zeros(image_size, dtype=np.uint8)

    for (y, x, c), piece in pieces:
        patch_h, patch_w = piece.shape[:2]
        if piece.ndim == 3 and piece.shape[2] > 1:
            reconstructed_image[y:y + patch_h, x:x + patch_w, :] = piece
        else:
            reconstructed_image[y:y + patch_h, x:x + patch_w, c:c + 1] = piece

    return reconstructed_image

def redraw_image(patch, reconstructed_image, piece_size=PIECE_SIZE):
    """

    only available for standard img, not for feature maps.
    """
    (y, x, c), piece = patch
    patch_h, patch_w = piece.shape[:2]

    if piece.ndim == 3 and piece.shape[2] > 1:
        reconstructed_image[y:y + patch_h, x:x + patch_w, :] = piece
    else:
        reconstructed_image[y:y + patch_h, x:x + patch_w, c:c + 1] = piece

    return reconstructed_image
