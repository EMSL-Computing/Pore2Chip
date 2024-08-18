# Description of the files in the tests folder

This readme summarizes and focuses on the functionalities of scripts testing various aspects of the Pore2Chip repository.

## Purpose and functionality of `test_coordination.py`

The Python script defines functions for generating a network and calculating coordination numbers, along with a main function to demonstrate their usage.

- The `generate_network` function creates a simple 3x3 cubic network with spheres and cylinders using `OpenPNM`. It prints the network information and returns the network as a dictionary.
- The `test_coordination_nums_3D` function takes a network dictionary and calculates coordination numbers in 3D using the `coordination_nums_3D` function. It prints the coordination numbers and their mean and returns the coordination numbers.
- The `test_coordination_nums_2D` function takes a list of 3D images (`img_list`) and calculates coordination numbers in 2D using the `coordination_nums_2D` function. It prints the coordination numbers and their mean and returns the coordination numbers.

## Purpose and functionality of `test_export.py`



## Purpose and functionality of `test_filter_im.py`



## Purpose and functionality of `test_flow.py`



## Purpose and functionality of `test_generate.py`



## Purpose and functionality of `test_metrics.py`