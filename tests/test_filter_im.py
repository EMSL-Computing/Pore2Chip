import sys
import os
import time
import numpy as np
import porespy as ps
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

#mod_path = Path("__file__").resolve().parents[2]
#sys.path.append(os.path.abspath(mod_path))

#import pore2chip
#from pore2chip.src.pore2chip.export import extract_diameters, feret_diameter
from pore2chip.filter_im import filter_single, filter_list

def test_filter(image):
    """
    Filters image using different parameters

    Returns:
        Filtered images: Dataset of 3 filtered images
    """
    filtered1 = filter_single(image, cropx=[50, 100], cropy=[50, 100])
    filtered2 = filter_single(image, thresh=50, invert=True)
    filtered3 = filter_single(image, gauss=7, invert=True)

    return filtered1, filtered2, filtered3

def test_filter_list(image_stack):
    """
    Filters a 3D array (3D stack of images)

    Returns:
        Filtered images: Dataset of filtered image stacks
    """
    filtered_stack = filter_list(image_stack, cropx=[0,50], cropy=[25,80])
    return filtered_stack

def main():
    # Generate a test image using PoreSpy
    test_image = ps.generators.fractal_noise([500, 500], seed=1)
    test_image2 = cv2.normalize(test_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    filtered1, filtered2, filtered3 = test_filter(test_image2)

    # Display original and 3 filtered images
    fig, ax = plt.subplots(1,4)
    ax[0].imshow(test_image)
    ax[1].imshow(filtered1)
    ax[2].imshow(filtered2)
    ax[3].imshow(filtered3)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    ### Advanced Tests ###

    # Generate a 3D stack of images using PoreSpy
    test_stack = ps.generators.fractal_noise([3, 500, 500], seed=1)
    test_stack2 = cv2.normalize(test_stack, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    filtered_stack = test_filter_list(test_stack2)

    fig, ax = plt.subplots(1,4)
    ax[0].imshow(test_stack[0])
    ax[1].imshow(filtered_stack[0])
    ax[2].imshow(filtered_stack[1])
    ax[3].imshow(filtered_stack[2])
    plt.show()
    plt.waitforbuttonpress()
    plt.close()


if __name__ == "__main__":
    main()