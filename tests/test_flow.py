import sys
import os
import time
import numpy as np
from skimage.draw import ellipse

import flow.pinn_utilities
import flow.plotting_results


def test_plot_xct(image):
    """
    Plots 2D XCT data segment using the flow.plotting_results module.

    Args:
        image (np.ndarray): The image data to plot.

    This function does not return anything but saves the plotted results to a specified directory.
    """
    results_dir = "../" # Adjust directory path as needed
    flow.plotting_results.plot_2d_xct_data_seg(image, results_dir=results_dir)


def test_generate_boundary(image):
    """
    Placeholder function for generating boundary conditions on the image.

    Args:
        image (np.ndarray): The image to process for boundary generation.

    Returns:
        None:.
    """

    # Currently, this function does nothing but is a placeholder for future boundary processing complex logic.
    return


def main():
    """
    Main function to generate a test image and apply plotting and boundary generation functions.

    This function orchestrates the process of creating an image, modifying it with geometric shapes,
    and applying visualization and analysis functions.
    """

    # Generate a test image using skimage.draw
    test_image = np.zeros((500, 500), dtype=np.uint8)

    # Draw several ellipses on the image at various positions and sizes
    rr, cc = ellipse(60, 60, 40, 60)
    test_image[rr, cc] = 1 # Set pixels inside the ellipse to white

    # Add more circular regions as needed
    rr, cc = ellipse(400, 400, 80, 60)
    test_image[rr, cc] = 1

    rr, cc = ellipse(200, 300, 100, 40)
    test_image[rr, cc] = 1

    rr, cc = ellipse(300, 200, 40, 100)
    test_image[rr, cc] = 1

    rr, cc = ellipse(400, 175, 100, 20)
    test_image[rr, cc] = 1

    rr, cc = ellipse(50, 300, 55, 20)
    test_image[rr, cc] = 1

    # Use the test image to demonstrate the plot function
    test_plot_xct(test_image)


if __name__ == "__main__":
    main()
