import hashlib

import numpy as np
from PIL import Image


def image_hash(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to a numpy array
        img_array = np.array(img)
        # Convert the numpy array to bytes
        img_bytes = img_array.tobytes()
        # Compute the hash of the image
        img_hash = hashlib.md5(img_bytes).hexdigest()
        return img_hash


def compare_images(image_path1, image_path2):
    hash1 = image_hash(image_path1)
    hash2 = image_hash(image_path2)
    return hash1 == hash2
