"""
feature_extraction.py
---------------------

Handcrafted feature extraction for the Nail Image Branch.

Extracted Features
------------------
1. RGB statistics
2. HSV statistics
3. GLCM texture features

The resulting feature vector will later be used by
the Random Forest classifier.

"""

import cv2
import numpy as np

from skimage.feature import graycomatrix
from skimage.feature import graycoprops

def rgb_features(image):
    """
    Compute mean and standard deviation
    of each RGB channel.

    Returns
    -------
    6 features
    """

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    features = []
    for channel in range(3):
        c = image[:, :, channel]
        features.append(np.mean(c))
        features.append(np.std(c))

    return features

def hsv_features(image):
    """
    Compute HSV statistics.

    Returns
    -------
    6 features
    """

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    features = []

    for channel in range(3):
        c = hsv[:, :, channel]
        features.append(np.mean(c))
        features.append(np.std(c))

    return features

def gray_image(image):
    """
    Convert image to grayscale.
    """

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    return gray

def compute_glcm(gray):
    """
    Compute Gray-Level Co-occurrence Matrix.

    Distance = 1 pixel

    Angles

    0°

    45°

    90°

    135°
    """

    matrix = graycomatrix(
        gray,
        distances=[1],
        angles=[
            0,
            np.pi/4,
            np.pi/2,
            3*np.pi/4
        ],

        symmetric=True,
        normed=True
    )

    return matrix

def glcm_features(image):
    """
    Extract texture statistics.

    Features

    Contrast

    Correlation

    Energy

    Homogeneity

    Returns
    -------

    16 values
    """

    gray = gray_image(image)
    matrix = compute_glcm(gray)
    features = []
    properties = [

        "contrast",

        "correlation",

        "energy",

        "homogeneity"

    ]
    for prop in properties:
        values = graycoprops(
            matrix,
            prop
        )[0]

        features.extend(values.tolist())

    return features

def color_features(image):
    """
    Combined color descriptor.

    Returns

    RGB

    +

    HSV
    """

    rgb = rgb_features(image)
    hsv = hsv_features(image)

    return rgb + hsv

def texture_features(image):
    """
    Combined texture descriptor.

    Returns

    GLCM features
    """

    return glcm_features(image)

def feature_vector_length():
    """
    Number of handcrafted features
    generated.
    """

    return len(feature_names())

from skimage.feature import local_binary_pattern

# def lbp_features(image):
#     """
#     Compute Local Binary Pattern histogram.

#     Returns
#     -------
#     10-bin normalized histogram.
#     """

#     gray = gray_image(image)

#     radius = 2
#     points = 8 * radius

#     lbp = local_binary_pattern(
#         gray,
#         points,
#         radius,
#         method="uniform"
#     )

#     hist, _ = np.histogram(
#         lbp.ravel(),
#         bins=np.arange(0, points + 4),
#         range=(0, points + 3)
#     )

#     hist = hist.astype(np.float32)
#     hist /= (hist.sum() + 1e-8)

#     return hist.tolist()

def lbp_features(image):
    """
    Compute Local Binary Pattern histogram.

    Returns
    -------
    10-bin normalized histogram.
    """

    gray = gray_image(image)

    radius = 2
    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=10,
        range=(0, points + 2)
    )

    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)

    return hist.tolist()

def shape_features(image):
    """
    Extract simple geometric descriptors.

    Returns
    -------
    Area
    Perimeter
    Aspect Ratio
    Circularity
    Solidity
    """

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return [0, 0, 0, 0, 0]

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / (h + 1e-8)

    circularity = (
        4 * np.pi * area /
        (perimeter ** 2 + 1e-8)
    )

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-8)

    return [
        area,
        perimeter,
        aspect_ratio,
        circularity,
        solidity
    ]

def extract_features(image):
    """
    Generate complete handcrafted
    feature vector.
    """

    features = []

    features.extend(color_features(image))
    features.extend(texture_features(image))
    features.extend(lbp_features(image))
    features.extend(shape_features(image))

    return np.array(
        features,
        dtype=np.float32
    )

def feature_names():
    """
    Names corresponding to each feature.

    Useful for Random Forest
    feature importance plots.
    """

    names = []

    rgb = [
        "R_mean","R_std",
        "G_mean","G_std",
        "B_mean","B_std"
    ]

    hsv = [
        "H_mean","H_std",
        "S_mean","S_std",
        "V_mean","V_std"
    ]

    names.extend(rgb)
    names.extend(hsv)

    props = [
        "contrast",
        "correlation",
        "energy",
        "homogeneity"
    ]

    angles = [
        "0",
        "45",
        "90",
        "135"
    ]

    for p in props:
        for a in angles:
            names.append(f"{p}_{a}")

    for i in range(10):
        names.append(f"LBP_{i}")

    names.extend([
        "Area",
        "Perimeter",
        "AspectRatio",
        "Circularity",
        "Solidity"
    ])

    return names

if __name__ == "__main__":

    from preprocess import preprocess_image
    from segment import segment_nail

    IMAGE = "../../data/nail_dataset/Normal/ID001.jpg"

    img = preprocess_image(IMAGE)
    nail = segment_nail(img)

    rgb = rgb_features(nail)
    hsv = hsv_features(nail)
    glcm = glcm_features(nail)
    print()
    print("RGB Features")
    print(rgb)
    print()
    print("HSV Features")
    print(hsv)
    print()    
    print("GLCM Features")
    print(glcm)
    print()    
    print("Feature Length")
    print(feature_vector_length())

    features = extract_features(nail)
    print()
    print("Total Features")
    print(len(features))
    print()
    print(feature_names())
    print()

    print(features)