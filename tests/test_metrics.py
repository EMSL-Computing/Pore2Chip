import sys
import os
import time
import numpy as np
import porespy as ps  # ✅ Add this line to fix the NameError
from skimage.draw import ellipse
from pathlib import Path
from porespy.generators import cylinders
import matplotlib.pyplot as plt

# Uncomment and modify these lines if the pore2chip package is not in the PYTHONPATH
#mod_path = Path("__file__").resolve().parents[2]
#sys.path.append(os.path.abspath(mod_path))

import pore2chip
from pore2chip.metrics import extract_diameters, feret_diameter, extract_diameters_alt, extract_diameters2


def test_feret_diameter(test_image):
    """
    Extracts feret diameters on test image and compares the 
    results to the pore diameters extracted from before

    Args:
        test_image (np.ndarray): The image from which to extract feret diameters.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing arrays of min feret sizes and 
            max sizes respectively
            - min_feret_diameters: Array of minimum Feret diameters.
            - max_feret_diameters: Array of maximum Feret diameters.
    """
    f_diameters = feret_diameter(test_image)
    return f_diameters


def test_extract_diameters(image, alt=False):
    """
    Extracts pore sizes and pore throat sizes on test image

    Args:
        image (np.ndarray): The image from which to extract diameters.
        alt (bool, optional): Flag to decide whether to use an alternative 
                              method for diameter extraction.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing arrays of pore sizes and 
            throat sizes respectively
            - pore_sizes: Array of pore sizes.
            - throat_sizes: Array of throat sizes.
    """
    if alt:
        diameters = extract_diameters_alt(image)
    else:
        diameters = extract_diameters(image)

    return diameters


def test_extract_diameters2(image):
    """
    Extracts pore sizes and pore throat sizes on test image
    This one used the skimage segmentation directly instead
    of using PoreSpy's built in SNOW implementation
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing arrays of pore sizes and 
            throat sizes respectively
            - pore_sizes: Array of pore sizes.
            - throat_sizes: Array of throat sizes.
    """
    diameters = extract_diameters2(image)
    return diameters


def main():
    """
    Main function to generate test images, extract pore/throat sizes, 
    apply diameter extraction methods, and display results.
    """

    # Generate a test image using skimage.draw
    test_image = np.zeros((500, 500), dtype=np.uint8)
    rr, cc = ellipse(60, 60, 40, 60)
    test_image[rr, cc] = 1
    rr, cc = ellipse(400, 400, 80, 60)
    test_image[rr, cc] = 1
    rr, cc = ellipse(200, 300, 100, 40)
    test_image[rr, cc] = 1
    rr, cc = ellipse(300, 200, 40, 100)
    test_image[rr, cc] = 1

    # Apply diameter extraction and print results
    diameters = test_extract_diameters(test_image)

    f_diameters = test_feret_diameter(test_image)

    print(f_diameters)
    print(diameters)
    print(diameters[0].min())
    print(diameters[0].max())

    ### Advanced Tests ###

    # Generate 3D cylinders
    cylinder_3D = cylinders([3, 500, 500], r=10, ncylinders=100, seed=1)

    #fig, ax = plt.subplots()
    #ax.imshow(cylinder_3D[0])
    #plt.show()

    diameters = test_extract_diameters2(cylinder_3D)

    diameters2 = test_extract_diameters(cylinder_3D, alt=True)

    print(diameters[0])
    print(diameters2)


if __name__ == "__main__":
    main()
