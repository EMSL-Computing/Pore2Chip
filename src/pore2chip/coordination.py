"""
Functions to calculate coordination number for 2D images and 3D image stack.
"""

import numpy as np
import porespy as ps
import openpnm as op
from skimage.segmentation import watershed
import scipy.ndimage as spim
from porespy.tools import randomize_colors


def coordination_nums_3D(img_list=None, pn=None, alt=True):
    r"""
      This function calculates the coordination number of each pore in a 3D image.

      **Parameters:**

      * `img_list` (list): A list of 3D images represented as NumPy arrays.
      * `pn` (openpnm.network.Network, optional): An optional OpenPNM network object. 
          If provided, the function will directly calculate coordination numbers from the network
          instead of processing images.

      **Returns:**

      * `coordination_nums` (list): A list containing the coordination number for each pore in the image(s).

      **Functionality:**

      1. **Check for OpenPNM network:**
         - If `pn` is not None (i.e., an OpenPNM network is provided), the function extracts the coordination numbers
           directly from the network using `op.models.network.coordination_number(pn)`.
         - This bypasses the image processing step and leverages the existing information in the network.

      2. **Image processing (if no network provided):**
         - If `pn` is None, the function assumes `img_list` contains 3D image data as NumPy arrays.
         - It utilizes the `porespy` library to process the images:
            - `ps.networks.snow2(img_list, voxel_size=1)`: This function likely performs segmentation and network extraction from the images.
              - `voxel_size=1` is assumed to be a unit voxel size (adjust if needed).
         - The processed network is then converted to an OpenPNM network using `op.io.network_from_porespy(snow_output.network)`.

      3. **Coordination number calculation:**
         - Regardless of the input (image or network), the function calculates the coordination number of each pore using:
            - `op.models.network.coordination_number(pn2)`: This function calculates the coordination number for each pore in the OpenPNM network `pn2`.

      4. **Return coordination numbers:**
         - The function returns the calculated coordination numbers (`coordination_nums`) as a list.

      **Notes:**

      - This function assumes the `porespy` and `openpnm` libraries are installed and imported.
      - The specific implementation of `ps.networks.snow2` might require further investigation depending on the exact functionality of the PoresPy library.
    """

    if pn is not None:
        coordination_nums = op.models.network.coordination_number(pn)
        return coordination_nums
    elif img_list is not None:
        if alt:
            sigma = 0.4
            dt = spim.distance_transform_edt(input=img_list)
            dt1 = spim.gaussian_filter(input=dt, sigma=sigma)
            peaks = ps.filters.find_peaks(dt=dt)

            #print('Initial number of peaks: ', spim.label(peaks)[1])
            peaks = ps.filters.trim_saddle_points(peaks=peaks, dt=dt1)
            #print('Peaks after trimming saddle points: ', spim.label(peaks)[1])
            peaks = ps.filters.trim_nearby_peaks(peaks=peaks, dt=dt)
            peaks, N = spim.label(peaks)
            #print('Peaks after trimming nearby peaks: ', N)

            regions = watershed(image=-dt, markers=peaks, mask=dt > 0)
            regions = randomize_colors(regions)
            net = ps.networks.regions_to_network(regions*img_list, voxel_size=1)
            pn = op.io.network_from_porespy(net)
            coordination_nums = op.models.network.coordination_number(pn)
            return coordination_nums
        else:
            snow_output = ps.networks.snow2(img_list, voxel_size=1)
            pn2 = op.io.network_from_porespy(snow_output.network)
            coordination_nums = op.models.network.coordination_number(pn2)

            return coordination_nums
    else:  
        return None


def coordination_nums_2D(img_list):
    r"""
      This function calculates the coordination number of each pore in a series of 2D images.

      **Parameters:**

      * `img_list` (list): A list of 2D images represented as NumPy arrays. Each image is assumed to be a single slice from a 3D volume.

      **Returns:**

      * `coordination_nums_2D` (list): A list containing the coordination number for each pore across all the 2D images.

      **Functionality:**

      1. **Iterate through image slices:**
         - The function loops through each image in `img_list` using a for loop.

      2. **Process individual image:**
         - Inside the loop, for each image slice (`img_list[k]`), it extracts a 2D portion using slicing (`img_list[k, :, :]`).
         - `ps.networks.snow2(img_slice, voxel_size=1)` likely performs segmentation and network extraction specifically for the 2D slice.
            - `voxel_size=1` is assumed to be a unit voxel size (adjust if needed).
         - The processed network information is stored in `snow_output.network`.
         - An OpenPNM network object (`pn`) is created by converting the PoresPy network using `op.io.network_from_porespy(snow_output.network)`.

      3. **Coordination number calculation:**
         - The coordination number for each pore in the current image slice's network (`pn`) is calculated using `op.models.network.coordination_number(pn)`.
         - This function from OpenPNM provides the coordination numbers as a list.

      4. **Accumulate coordination numbers:**
         - The calculated coordination numbers for the current slice (`temp_coordination`) are appended to the `coordination_nums_2D` list using `np.concatenate`.
         - This accumulates coordination numbers from all processed slices into a single list.

      5. **Return results:**
         - After iterating through all images, the function returns the final `coordination_nums_2D` list containing coordination numbers for all pores across the entire 2D image series.

      **Notes:**

      - This function assumes the `porespy` and `openpnm` libraries are installed and imported.
      - The specific implementation of `ps.networks.snow2` might require further investigation depending on the exact functionality of the PoresPy library.
    """

    coordination_nums_2D = []
    #
    if img_list.ndim == 2:
        snow_output = ps.networks.snow2(img_list, voxel_size=1)
        pn = op.io.network_from_porespy(snow_output.network)
        temp_coordination = op.models.network.coordination_number(pn)

        return temp_coordination

    else:
        for k in range(len(img_list)):
           # Use the Snow algorithm (included in PoreSpy) to calculate a pore
           # network and convert it to an OpenPNM network
           snow_output = ps.networks.snow2(img_list[k, :, :], voxel_size=1)
           pn = op.io.network_from_porespy(snow_output.network)
           #
           temp_coordination = op.models.network.coordination_number(pn)
           coordination_nums_2D = np.concatenate(
              (coordination_nums_2D, temp_coordination))

        return coordination_nums_2D
