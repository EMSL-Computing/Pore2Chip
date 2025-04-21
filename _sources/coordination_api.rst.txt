
**coordination**
================

The ``coordination`` module contains functions related to extracting pore coordination numbers.

----

coordination_nums_3D()
----------------------

.. autofunction:: pore2chip.coordination.coordination_nums_3D

Functionality:

1. Check for OpenPNM network:
    - If `pn` is not None (i.e., an OpenPNM network is provided), the function extracts the coordination numbers directly from the network using `op.models.network.coordination_number(pn)`.
    - This bypasses the image processing step and leverages the existing information in the network.

2. Image processing (if no network provided):
    - If `pn` is None, the function assumes `img_list` contains 3D image data as NumPy arrays.
    - It utilizes the `porespy` library to process the images:
    - `ps.networks.snow2(img_list, voxel_size=1)`: This function likely performs segmentation and network extraction from the images.
        - `voxel_size=1` is assumed to be a unit voxel size (adjust if needed).
    - The processed network is then converted to an OpenPNM network using `op.io.network_from_porespy(snow_output.network)`.

3. Coordination number calculation:
    - Regardless of the input (image or network), the function calculates the coordination number of each pore using:
    - `op.models.network.coordination_number(pn2)`: This function calculates the coordination number for each pore in the OpenPNM network `pn2`.

4. Return coordination numbers:
    - The function returns the calculated coordination numbers (`coordination_nums`) as a list.

..

    - This function assumes the `porespy` and `openpnm` libraries are installed and imported.
    - The specific implementation of `ps.networks.snow2` might require further investigation depending on the exact functionality of the PoresPy library.

----

coordination_nums_2D()
----------------------

.. autofunction:: pore2chip.coordination.coordination_nums_2D

Functionality:

1. Iterate through image slices:
    - The function loops through each image in `img_list` using a for loop.

2. Process individual image:
    - Inside the loop, for each image slice (`img_list[k]`), it extracts a 2D portion using slicing (`img_list[k, :, :]`).
    - `ps.networks.snow2(img_slice, voxel_size=1)` likely performs segmentation and network extraction specifically for the 2D slice.
    - `voxel_size=1` is assumed to be a unit voxel size (adjust if needed).
    - The processed network information is stored in `snow_output.network`.
    - An OpenPNM network object (`pn`) is created by converting the PoresPy network using `op.io.network_from_porespy(snow_output.network)`.

3. Coordination number calculation:
    - The coordination number for each pore in the current image slice's network (`pn`) is calculated using `op.models.network.coordination_number(pn)`.
    - This function from OpenPNM provides the coordination numbers as a list.

4. Accumulate coordination numbers:
    - The calculated coordination numbers for the current slice (`temp_coordination`) are appended to the `coordination_nums_2D` list using `np.concatenate`.
    - This accumulates coordination numbers from all processed slices into a single list.

5. Return results:
    - After iterating through all images, the function returns the final `coordination_nums_2D` list containing coordination numbers for all pores across the entire 2D image series.

..

    - This function assumes the `porespy` and `openpnm` libraries are installed and imported.
    - The specific implementation of `ps.networks.snow2` might require further investigation depending on the exact functionality of the PoresPy library.