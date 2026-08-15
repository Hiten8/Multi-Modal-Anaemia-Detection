"""
segment.py
----------

Nail segmentation module.

This module extracts the finger region from the
background and prepares the ROI for nail extraction.

Pipeline

Original Image
        ↓
HSV Conversion
        ↓
Background Removal
        ↓
Morphological Cleaning
        ↓
Largest Contour
        ↓
Finger ROI

The ROI is then used by the nail extraction stage.
"""

import cv2
import numpy as np

def rgb_to_hsv(image):
    """
    Convert RGB image to HSV.
    """
    # If already uint8, leave it unchanged.
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    return hsv

def segment_finger(image):
    """
    Separate the finger from the dark background.

    Returns
    -------
    Binary mask
    """

    hsv = rgb_to_hsv(image)

    lower = np.array([0, 15, 20])

    upper = np.array([180, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    return mask

def clean_mask(mask):
    """
    Remove noise from binary mask.
    """

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask

def largest_contour(mask):
    """
    Returns the largest contour
    in the binary image.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    return contour

def contour_mask(mask):
    """
    Create a clean mask from
    largest contour.
    """

    contour = largest_contour(mask)

    if contour is None:

        return np.zeros_like(mask)

    output = np.zeros_like(mask)

    cv2.drawContours(
        output,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return output

def finger_bounding_box(mask):
    """
    Compute bounding rectangle.

    Returns

    x
    y
    width
    height
    """

    contour = largest_contour(mask)

    if contour is None:

        return None

    x, y, w, h = cv2.boundingRect(contour)

    return x, y, w, h

def crop_finger(image,
                mask):
    """
    Crop the finger region.

    Returns

    Cropped image
    """

    box = finger_bounding_box(mask)

    if box is None:

        return image

    x, y, w, h = box

    padding = 10

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    cropped = image[
        y1:y2,
        x1:x2
    ]

    return cropped

def apply_mask(image,
               mask):
    """
    Remove background.
    """

    image_uint8 = (image * 255).astype(
        np.uint8
    )

    output = cv2.bitwise_and(
        image_uint8,
        image_uint8,
        mask=mask
    )

    output = output.astype(
        np.float32
    ) / 255.0

    return output

def extract_finger(image):
    """
    Complete finger segmentation.

    Returns

    finger image

    finger mask
    """

    mask = segment_finger(image)

    mask = clean_mask(mask)

    mask = contour_mask(mask)

    finger = apply_mask(
        image,
        mask
    )

    finger = crop_finger(
        finger,
        mask
    )

    return finger, mask

def locate_nail_region(finger):
    """
    Estimate nail location.

    Assumption:
    Nail occupies the upper
    35-45% of the finger ROI.
    """

    h, w = finger.shape[:2]

    top = 0
    bottom = int(h * 0.45)

    nail = finger[top:bottom, :]

    return nail

def nail_mask(nail):
    """
    Build binary mask for nail.

    Nails are generally brighter than
    surrounding skin.
    """

    if nail.dtype != np.uint8:
        nail = (nail * 255).astype(np.uint8)

    gray = cv2.cvtColor(
        nail,
        cv2.COLOR_RGB2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3,3),np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask

def extract_nail(nail,
                 mask):
    """
    Extract largest connected component.
    """

    contour = largest_contour(mask)

    if contour is None:
        return nail

    x,y,w,h = cv2.boundingRect(contour)

    crop = nail[
        y:y+h,
        x:x+w
    ]

    return crop

def resize_nail(
        nail,
        size=(224,224)):
    """
    Standardize nail size.
    """

    return cv2.resize(
        nail,
        size,
        interpolation=cv2.INTER_AREA
    )

def segment_nail(image):
    """
    Complete segmentation pipeline.

    Parameters
    ----------

    RGB image

    Returns
    -------

    Segmented nail image
    """

    finger, finger_mask = extract_finger(image)

    nail_roi = locate_nail_region(finger)

    mask = nail_mask(nail_roi)

    nail = extract_nail(
        nail_roi,
        mask
    )

    nail = resize_nail(nail)

    return nail

def visualize_segmentation(image):
    """
    Display every stage
    of the segmentation pipeline.
    """

    import matplotlib.pyplot as plt

    finger, finger_mask = extract_finger(image)

    nail_roi = locate_nail_region(finger)

    mask = nail_mask(nail_roi)

    nail = extract_nail(
        nail_roi,
        mask
    )

    plt.figure(figsize=(14,4))

    plt.subplot(141)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(142)
    plt.imshow(finger)
    plt.title("Finger")
    plt.axis("off")

    plt.subplot(143)
    plt.imshow(mask,cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(144)
    plt.imshow(nail)
    plt.title("Segmented Nail")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from preprocess import preprocess_image

    IMAGE = "../../data/nail_dataset/Normal/ID001.jpg"

    img = preprocess_image(IMAGE)

    visualize_segmentation(img)

    finger, mask = extract_finger(img)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(mask, cmap="gray")
    plt.title("Finger Mask")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(finger)
    plt.title("Finger ROI")
    plt.axis("off")

    plt.show()