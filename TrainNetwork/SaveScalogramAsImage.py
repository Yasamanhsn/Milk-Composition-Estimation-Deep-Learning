import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_scalogram_as_image(sig_1, sig_2, folder_path, index):
    """
    Save a combined scalogram image from two input signals as a JPEG file.

    Parameters:
    sig_1 (numpy.ndarray): First signal's scalogram.
    sig_2 (numpy.ndarray): Second signal's scalogram.
    folder_path (str): Path to the folder where the image will be saved.
    index (int): Index for naming the image file.

    Returns:
    None
    """
    image_size = [224, 224]

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Stack signals vertically to create the combined scalogram
    scalogram = np.vstack((sig_1, sig_2))

    # resize the original image to the desire
    scalogram_resize = cv2.resize(scalogram, image_size)
    # Normalize the combined scalogram and convert it to a colorful representation
    scalogram_resize = cv2.normalize(scalogram_resize, None, 0, 255, cv2.NORM_MINMAX)
    scalogram_resize = np.uint8(scalogram_resize)
    scalogram_resize = cv2.applyColorMap(scalogram_resize, cv2.COLORMAP_JET)

    # Save the image as a JPEG file
    image_path = os.path.join(folder_path, f"{index}.jpg")
    cv2.imwrite(image_path, scalogram_resize)





