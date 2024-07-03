import sys
import os
import time
import numpy as np
from skimage.draw import ellipse


import flow.pinn_utilities
import flow.plotting_results

def test_plot_xct(image):
    results_dir = "../"
    flow.plotting_results.plot_2d_xct_data_seg(image, results_dir=results_dir)

def test_generate_boundary(image):
    return

def main():
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
    rr, cc = ellipse(400, 175, 100, 20)
    test_image[rr, cc] = 1
    rr, cc = ellipse(50, 300, 55, 20)
    test_image[rr, cc] = 1

    test_plot_xct(test_image)


if __name__ == "__main__":
    main()