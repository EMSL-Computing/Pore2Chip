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

      Args:
         img_list (list): A list of 3D images represented as NumPy arrays.
         pn (openpnm.network.Network): An optional OpenPNM network object. 
            If provided, the function will directly calculate coordination numbers from the network
            instead of processing images.

      Returns:
         list: A list containing the coordination number for each pore in the image(s).
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
            net = ps.networks.regions_to_network(regions * img_list,
                                                 voxel_size=1)
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

      Args:
         img_list (list): A list of 2D images represented as NumPy arrays. Each image is assumed to be a single slice from a 3D volume.

      Returns:
         list: A list containing the coordination number for each pore across all the 2D images.

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
