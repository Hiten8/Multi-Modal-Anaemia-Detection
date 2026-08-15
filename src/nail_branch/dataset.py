"""
dataset.py
----------

Dataset loader for the Nail Image Branch.

Responsibilities
----------------
1. Traverse dataset folders
2. Read every image
3. Assign labels
4. Call preprocessing
5. Call segmentation
6. Call feature extraction
7. Return feature matrix and labels

Dataset Structure
-----------------

data/
    nail_dataset/
        Normal/
        Leve/
        Moderada/
"""

import os

import cv2
import numpy as np

from preprocess import preprocess_image
from segment import segment_nail
from feature_extraction import extract_features

LABELS = {

    "Normal": 0,
    "Leve": 1,
    "Moderada": 2
}

IMAGE_EXTENSIONS = (

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

def is_image_file(filename):
    """
    Check whether file is an image.
    """

    return filename.lower().endswith(
        IMAGE_EXTENSIONS
    )

def class_folders(dataset_path):
    """
    Returns dictionary

    {
        class_name : folder_path
    }
    """

    folders = {}

    for label in LABELS:

        folder = os.path.join(
            dataset_path,
            label
        )

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Missing folder : {folder}"
            )

        folders[label] = folder

    return folders

def image_paths(folder):
    """
    Returns all image paths
    inside a folder.
    """

    paths = []

    for file in sorted(os.listdir(folder)):

        if is_image_file(file):

            paths.append(

                os.path.join(
                    folder,
                    file
                )

            )

    return paths

def dataset_summary(dataset_path):
    """
    Print number of images
    in every class.
    """

    folders = class_folders(
        dataset_path
    )

    print()
    print("=" * 40)
    print("Dataset Summary")
    print("=" * 40)

    total = 0

    for label, folder in folders.items():

        count = len(
            image_paths(folder)

        )
        total += count
        print(
            f"{label:10s}: {count}"
        )
    print()
    print(
        f"Total Images : {total}"
    )
    print("=" * 40)
    print()

def process_image(image_path):
    """
    Complete pipeline for one image.

    Image

    ↓

    Preprocess

    ↓

    Segment Nail

    ↓

    Extract Features
    """

    image = preprocess_image(
        image_path
    )

    nail = segment_nail(
        image
    )

    features = extract_features(
        nail
    )

    return features

def initialize_dataset():
    """
    Creates empty feature
    and label lists.
    """

    X = []
    y = []
    filenames = []

    return X, y, filenames

def encode_label(class_name):
    """
    Convert class name
    into integer.
    """

    return LABELS[class_name]

def process_class(folder, label, X, y, filenames):
    """
    Process all images
    in one class folder.
    """

    paths = image_paths(
        folder
    )
    print()
    print(f"Processing {label}")

    for path in paths:

        try:
            features = process_image(path)
            if features is None or len(features) == 0:
                continue
            X.append(features)
            y.append(
                encode_label(label)
            )
            filenames.append(
                os.path.basename(path)
            )
        except Exception as e:
            print()
            print(
                f"Skipped {path}"
            )
            print(e)

    return X, y, filenames

def shuffle_dataset(X, y, filenames=None, random_state=42):
    """
    Shuffle dataset while preserving
    correspondence between features,
    labels and filenames.
    """

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    if filenames is not None:
        filenames = np.array(filenames)[indices]

    return X, y, filenames

def save_features(X, y, output_folder, filenames=None):
    """
    Save extracted features.
    """
    os.makedirs(output_folder, exist_ok=True)
    np.save(
        os.path.join(output_folder, "X.npy"),
        X
    )
    np.save(
        os.path.join(output_folder, "y.npy"),
        y
    )
    if filenames is not None:
        np.save(
            os.path.join(output_folder, "filenames.npy"),
            filenames
        )
    print()
    print(f"Saved features to {output_folder}")

def load_saved_features(output_folder):
    """
    Load saved features.
    """

    X = np.load(
        os.path.join(output_folder, "X.npy")
    )

    y = np.load(
        os.path.join(output_folder, "y.npy")
    )

    filenames_path = os.path.join(
        output_folder,
        "filenames.npy"
    )

    filenames = None

    if os.path.exists(filenames_path):
        filenames = np.load(
            filenames_path,
            allow_pickle=True
        )

    return X, y, filenames

def load_dataset(dataset_path, shuffle=True):
    """
    Load complete dataset.

    Returns

    X
    y
    filenames
    """

    dataset_summary(dataset_path)
    folders = class_folders(dataset_path)
    X, y, filenames = initialize_dataset()

    for label, folder in folders.items():
        X, y, filenames = process_class(
            folder,
            label,
            X,
            y,
            filenames
        )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    filenames = np.array(filenames)

    if shuffle:
        X, y, filenames = shuffle_dataset(
            X,
            y,
            filenames
        )

    return X, y, filenames

def dataset_statistics(y):
    """
    Print class distribution.
    """

    print()
    print("=" * 40)
    print("Class Distribution")
    print("=" * 40)

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    reverse_labels = {
        v: k
        for k, v in LABELS.items()

    }

    for cls, count in zip(unique, counts):
        print(
            f"{reverse_labels[cls]:10s}: {count}"
        )
    print("=" * 40)
    print()

if __name__ == "__main__":
    DATASET = "../../data/nail_dataset"
    OUTPUT = "../../outputs/features"
    X, y, filenames = load_dataset(DATASET)
    dataset_statistics(y)

    save_features(
        X,
        y,
        OUTPUT,
        filenames
    )

    print()
    print("Feature Matrix Shape")
    print(X.shape)

    print()
    print("Labels Shape")
    print(y.shape)

    print()
    print("First Feature Vector")
    print(X[0])

    print()
    print("First Label")
    print(y[0])