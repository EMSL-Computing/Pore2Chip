# Description of the files in the tests folder

This readme summarizes and focuses on the functionalities of scripts testing various aspects of the Pore2Chip repository.

## Purpose and functionality of `test_coordination.py`

The Python script defines functions for generating a network and calculating coordination numbers, along with a main function to demonstrate their usage.

- The `generate_network` function creates a simple 3x3 cubic network with spheres and cylinders using `OpenPNM`. It prints the network information and returns the network as a dictionary.
- The `test_coordination_nums_3D` function takes a network dictionary and calculates coordination numbers in 3D using the `coordination_nums_3D` function. It prints the coordination numbers and their mean and returns the coordination numbers.
- The `test_coordination_nums_2D` function takes a list of 3D images (`img_list`) and calculates coordination numbers in 2D using the `coordination_nums_2D` function. It prints the coordination numbers and their mean and returns the coordination numbers.

## Purpose and functionality of `test_export.py`

This python script through `test_network2svg` and `test_network2dxf` functions directly test the capability of the `network2svg` and `network2dxf` functions from the `pore2chip` package to handle the generated network structure and produce relevant output files for computational simulations and visualizations. This allows us to verify that the generated SVG and DXF files visually match the expected output for a 3x3 grid of circles connected by straight lines (throats). That is, it generates a sample network. Calls functions to convert the network to SVG and DXF formats, which is essential for experimental design testing.

- The `generate_network` creates a hardcoded network structure with 3x3 pores and straight throats. It returns a dictionary containing pore coordinates, diameters, throat connections, and diameters.
- The `test_network2svg` converts the given network to an SVG using network2svg, saves the SVG as `grain_network.svg`, and then converts the SVG to a PNG using `cairosvg` for visualization
- The `test_network2dxf` converts the given network to a DXF using `network2dxf` and then saves the DXF as `test_dxf_1.dxf`. 

## Purpose and functionality of `test_filter_im.py`

This python script applies applies image filters and displays the results for both 2D and 3D cases.

- The `test_filter` demonstrates how to apply various filters and cropping to an image.
- The `test_filter_list` applies a single type of filter across a stack of images, demonstrating handling of 3D data

## Purpose and functionality of `test_flow.py`

This python script provides simple test case (`test_plot_xct`) for visualizing xct intensity results. For more detailed and complex example, please see the Jupyter notebooks `example_6_flow_2d_numerical_on_XCT.ipynb` and `example_7_flow_2d_pinn_on_XCT.ipynb`

## Purpose and functionality of `test_generate.py`



## Purpose and functionality of `test_metrics.py`