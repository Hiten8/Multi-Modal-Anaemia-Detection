"""
preprocess.py
-------------

Image preprocessing utilities for the Nail Image Branch.

Pipeline

Image
   ↓
Resize
   ↓
RGB Conversion
   ↓
CLAHE
   ↓
Gaussian Blur
   ↓
Normalization

The output image is then passed to the segmentation module.
"""

import cv2
import numpy as np

IMAGE_SIZE = (224, 224)

def load_image(image_path: str):
    """
    Reads an image from disk.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    image : numpy.ndarray (RGB)
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def resize_image(image,
                 size=IMAGE_SIZE):
    """
    Resize image.

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    resized image
    """

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )

def apply_clahe(image):
    """
    Apply CLAHE only on the luminance channel.

    This improves contrast while preserving
    nail colour.

    Parameters
    ----------
    image : RGB image

    Returns
    -------
    Enhanced RGB image
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return enhanced

def remove_noise(image):
    """
    Mild Gaussian smoothing.

    Removes sensor noise while preserving
    nail edges.
    """

    return cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

def normalize(image):
    """
    Convert image into float32
    in the range [0,1].
    """

    image = image.astype(np.float32)

    image /= 255.0

    return image

def preprocess_image(image_path):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    Preprocessed RGB image
    """

    image = load_image(image_path)

    image = resize_image(image)

    image = apply_clahe(image)

    image = remove_noise(image)

    image = normalize(image)

    return image

def denormalize(image):
    """
    Convert normalized image back to uint8.
    Useful for visualization.
    """

    image = (image * 255).clip(
        0,
        255
    ).astype(np.uint8)

    return image

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    IMAGE = "../../data/nail_dataset/Normal/ID001.jpg"

    img = preprocess_image(IMAGE)

    plt.imshow(denormalize(img))
    plt.title("Preprocessed Image")
    plt.axis("off")
    plt.show()