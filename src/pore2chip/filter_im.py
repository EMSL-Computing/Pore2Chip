import numpy as np
import cv2 as cv
import copy
import os


def filter_single(image, 
                  cropx=None, 
                  cropy=None, 
                  grayMinimum=None, 
                  grayMaximum=None, 
                  thresh=None, 
                  gauss=5, 
                  invert=False):
    r"""
    Filters single image using OpenCV Gaussian Blur and thresholds images 
    using Otsu's thresholding.

    Args:
        image (2D array): Input image 
        cropx (tuple or similar): Number of pixels in x-axis to crop image by 
        cropy (tuple or similar): Number of pixels in y-axis to crop image by 
        grayMinimum (int): Minimum pixel value to count as solid 
        grayMaximum (int): Maximum pixel value to count as solid 
        thresh (int): Threshold value. If not given, then Otsu's threshold is used 
        gauss (int): Radius of Gaussian Blur. Default is 5 
        invert (Boolean): Inverts pixel values of image 

    Returns:
        2D numpy array: Filtered image
    """

    y_length = image.shape[0]
    x_length = image.shape[1]
    image_filtered = np.zeros((y_length, x_length), dtype=np.uint8)

    if cropx is not None and cropy is not None:
        y_length = cropy[1] - cropy[0]
        x_length = cropx[1] - cropx[0]
        image_filtered = copy.deepcopy(image[cropy[0]:cropy[1],
                                             cropx[0]:cropx[1]])
    else:
        image_filtered = copy.deepcopy(image)

    grayList = None
    if grayMinimum is not None and grayMaximum is not None:
        grayMin = grayMinimum  # 70
        grayMax = grayMaximum  # 90
        grayList = range(grayMin, grayMax)

    # Lighten some dark grays to separate them from the background and then
    # median blur
    if grayList is not None:
        image_filtered[np.isin(image_filtered, grayList)] = 200
        image_filtered = cv.medianBlur(image_filtered, 3)

    # Blur and Otsu's Threshold
    image_filtered = cv.GaussianBlur(image_filtered, (gauss, gauss), 0)

    if thresh is None:
        ret, image_filtered = cv.threshold(image_filtered, 0, 255,
                                           cv.THRESH_OTSU)
    else:
        ret, image_filtered = cv.threshold(image_filtered, thresh, 255,
                                           cv.THRESH_BINARY)

    # Inverted Version (PoreSpy and Skimage treats white pixels as pores, so
    # this makes an inverted copy of the original filtered list)
    if invert:
        image_filtered = cv.bitwise_not(image_filtered)

    return image_filtered


def filter_list(img_list,
                cropx=None,
                cropy=None,
                crop_depth=None,
                grayMinimum=None,
                grayMaximum=None,
                thresh=None,
                gauss=5,
                invert=False):
    r"""
    Filters array of images using OpenCV Gaussian Blur and thresholds images 
    using Otsu's thresholding.

    Args:
        img_list (3D numpy array): Array of input images
        cropx (tuple or similar): Number of pixels in x-axis to crop image by 
        cropy (tuple or similar): Number of pixels in y-axis to crop image by 
        crop_depth (int): Number of pixels in z-axis
        grayMinimum (int): Minimum pixel value to count as solid 
        grayMaximum (int): Maximum pixel value to count as solid 
        thresh (int): Threshold value. If not given, then Otsu's threshold is used
        gauss (int): Radius of Gaussian Blur. Default is 5.
        invert (Boolean): Inverts pixel values of image list

    Returns:
        3D numpy array : Array of filtered images
    """
    # 3D array setup
    depth = len(img_list)
    if crop_depth is not None:
        depth = crop_depth

    y_length = img_list[0, :, :].shape[0]
    x_length = img_list[0, :, :].shape[1]

    if cropx is not None and cropy is not None:
        y_length = cropy[1] - cropy[0]
        x_length = cropx[1] - cropx[0]

    image_stack_filtered = np.zeros((depth, y_length, x_length),
                                    dtype=np.uint8)

    grayList = None
    if grayMinimum is not None and grayMaximum is not None:
        grayMin = grayMinimum  # 70
        grayMax = grayMaximum  # 90
        grayList = range(grayMin, grayMax)

    for stride in range(depth):

        # Populate the filtered array by copying the original image to it
        if cropx is not None and cropy is not None:
            image_stack_filtered[stride, :, :] = copy.deepcopy(
                img_list[stride, cropy[0]:cropy[1], cropx[0]:cropx[1]])
        else:
            image_stack_filtered[stride, :, :] = copy.deepcopy(
                img_list[stride, :, :])

        # Lighten some dark grays to separate them from the background and then
        # median blur
        if grayList is not None:
            image_stack_filtered[stride, :, :][np.isin(
                image_stack_filtered[stride, :, :], grayList)] = 200
            image_stack_filtered[stride, :, :] = cv.medianBlur(
                image_stack_filtered[stride, :, :], 3)

        # Blur and Otsu's Threshold
        image_stack_filtered[stride, :, :] = cv.GaussianBlur(
            image_stack_filtered[stride, :, :], (gauss, gauss), 0)

        if thresh is None:
            ret, image_stack_filtered[stride, :, :] = cv.threshold(
                image_stack_filtered[stride, :, :], 0, 255, cv.THRESH_OTSU)
        else:
            ret, image_stack_filtered[stride, :, :] = cv.threshold(
                image_stack_filtered[stride, :, :], thresh, 255,
                cv.THRESH_BINARY)

        # Inverted Version (PoreSpy and Skimage treats white pixels as pores,
        # so this makes an inverted copy of the original filtered list)
        if invert:
            image_stack_filtered[stride, :, :] = cv.bitwise_not(
                image_stack_filtered[stride, :, :])

    return image_stack_filtered


def read_and_filter(img_path,
                    cropx=None,
                    cropy=None,
                    grayMinimum=None,
                    grayMaximum=None,
                    thresh=None,
                    gauss=5,
                    invert=False):
    r"""
    Reads and filters a single image using OpenCV Gaussian Blur and 
    thresholds images using Otsu's thresholding.

    Args:
        img_path (str): Absolute path to image
        cropx (tuple or similar): Number of pixels in x-axis to crop image by 
        cropy (tuple or similar): Number of pixels in y-axis to crop image by 
        grayMinimum (int): Minimum pixel value to count as solid 
        grayMaximum (int): Maximum pixel value to count as solid 
        thresh (int): Threshold value. If not given, then Otsu's threshold is used
        gauss (int): Radius of Gaussian Blur. Default is 5.
        invert (Boolean): Inverts pixel values of image list

    Returns:
        2D numpy array : Filtered image
    """
    image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    x_length = 100
    y_length = 100
    image_filtered = np.zeros((y_length, x_length), dtype=np.uint8)
    if cropx is not None and cropy is not None:
        y_length = cropy[1] - cropy[0]
        x_length = cropx[1] - cropx[0]
        image = copy.deepcopy(
            np.flipud(image[cropy[0]:cropy[1], cropx[0]:cropx[1]]))
    else:
        y_length = image.shape[0]
        x_length = image.shape[1]

    grayList = None
    if grayMinimum is not None and grayMaximum is not None:
        grayMin = grayMinimum  # 70
        grayMax = grayMaximum  # 90
        grayList = range(grayMin, grayMax)

    # Populate the filtered array by copying the original image to it
    image_filtered = copy.deepcopy(image)

    # Lighten some dark grays to separate them from the background and then
    # median blur
    if grayList is not None:
        image_filtered[np.isin(image_filtered, grayList)] = 200
        image_filtered = cv.medianBlur(image_filtered, 3)

    # Blur and Otsu's Threshold
    image_filtered = cv.GaussianBlur(image_filtered, (gauss, gauss), 0)
    ret3, image_filtered = cv.threshold(image_filtered, 0, 255, cv.THRESH_OTSU)

    # Inverted Version (PoreSpy and Skimage treats white pixels as pores, so
    # this makes an inverted copy of the original filtered list)
    if invert:
        image_filtered = cv.bitwise_not(image_filtered)

    return image_filtered


def read_and_filter_list(img_path,
                         cropx=None,
                         cropy=None,
                         crop_depth=None,
                         grayMinimum=None,
                         grayMaximum=None,
                         thresh=None,
                         gauss=5,
                         invert=False):
    r"""
    Reads and filters array of images using OpenCV Gaussian Blur and 
    thresholds images using Otsu's thresholding

    Args:
        img_path (str): Absolute path to directory containing images
        cropx (tuple or similar): Number of pixels in x-axis to crop image by 
        cropy (tuple or similar): Number of pixels in y-axis to crop image by 
        crop_depth (int): Number of pixels in z-axis 
        grayMinimum (int): Minimum pixel value to count as solid 
        grayMaximum (int): Maximum pixel value to count as solid 
        thresh (int): Threshold value. If not given, then Otsu's threshold is used
        gauss (int): Radius of Gaussian Blur. Default is 5.
        invert (Boolean): Inverts pixel values of image list

    Returns:
        3D numpy array : Array of filtered images
    """
    depth = 1
    if crop_depth is not None:
        depth = crop_depth
    else:
        depth = len(os.listdir(img_path))
    x_length = 100
    y_length = 100
    if cropx is not None and cropy is not None:
        y_length = cropy[1] - cropy[0]
        x_length = cropx[1] - cropx[0]
    else:
        test_img = cv.imread(img_path + os.listdir(img_path)[0],
                             cv.IMREAD_GRAYSCALE)
        y_length = test_img.shape[0]
        x_length = test_img.shape[1]
    image_list_3D = np.zeros((depth, y_length, x_length), dtype=np.uint8)

    # Filter Images
    image_list_3D_filtered = np.zeros((depth, y_length, x_length),
                                      dtype=np.uint8)
    grayList = None
    if grayMinimum is not None and grayMaximum is not None:
        grayMin = grayMinimum  # 70
        grayMax = grayMaximum  # 90
        grayList = range(grayMin, grayMax)

    for stride in range(depth):
        # Read Images
        img = cv.imread(img_path + os.listdir(img_path)[stride],
                        cv.IMREAD_GRAYSCALE)
        # Copy subsection of original image to 3D array
        if cropx is not None and cropy is not None:
            image_list_3D[stride, :, :] = copy.deepcopy(
                np.flipud(img[cropy[0]:cropy[1], cropx[0]:cropx[1]]))
        else:
            image_list_3D[stride, :, :] = copy.deepcopy(img)

        # Populate the filtered array by copying the original image to it
        image_list_3D_filtered[stride, :, :] = copy.deepcopy(
            image_list_3D[stride, :, :])

        # Lighten some dark grays to separate them from the background and then
        # median blur
        if grayList is not None:
            image_list_3D_filtered[stride, :, :][np.isin(
                image_list_3D_filtered[stride, :, :], grayList)] = 200
            image_list_3D_filtered[stride, :, :] = cv.medianBlur(
                image_list_3D_filtered[stride, :, :], 3)

        # Blur and Otsu's Threshold
        image_list_3D_filtered[stride, :, :] = cv.GaussianBlur(
            image_list_3D_filtered[stride, :, :], (gauss, gauss), 0)
        ret3, image_list_3D_filtered[stride, :, :] = cv.threshold(
            image_list_3D_filtered[stride, :, :], 0, 255, cv.THRESH_OTSU)

        # Inverted Version (PoreSpy and Skimage treats white pixels as pores,
        # so this makes an inverted copy of the original filtered list)
        if invert:
            image_list_3D_filtered[stride, :, :] = cv.bitwise_not(
                image_list_3D_filtered[stride, :, :])

    return image_list_3D_filtered
