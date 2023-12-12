import os
import cv2
import numpy as np
import pandas as pd

def load_images(data_folder, target_csv):
    """
    Load images and their corresponding target values from a specified folder and CSV file.

    Parameters:
    data_folder (str): Name of the folder containing image data.
    target_csv (str): Name of the CSV file containing target values.

    Returns:
    numpy.ndarray, numpy.ndarray: Arrays of loaded images and their target values.
    """
    root_addr = os.getcwd()
    data_dir = os.path.join(root_addr, data_folder)

    # Load target data from CSV file
    target_data = pd.read_csv(os.path.join(root_addr, target_csv))

    # Initialize lists to store image data and target values
    images = []
    targets = []

    num_im = len(os.listdir(data_dir))
    for i in range(num_im):
        img_path = os.path.join(data_dir, str(i + 1) + '.jpg')
        img = cv2.imread(img_path)  # Load image
        img = cv2.resize(img, (224, 224)) if img is not None else None  # Resize if necessary, handle None case
        if img is not None:
            images.append(img)

            # target data
            target = target_data.values[i,]
            targets.append(target)

    return np.array(images), np.array(targets)

