# Description of the files in the `tests` folder

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
- The `test_filter_list` applies a single type of filter across a stack of images, demonstrating handling of 3D data.

## Purpose and functionality of `test_flow.py`

This python script provides simple test case (`test_plot_xct`) for visualizing xct intensity results. For more detailed and complex example, please see the Jupyter notebooks `example_6_flow_2d_numerical_on_XCT.ipynb` and `example_7_flow_2d_pinn_on_XCT.ipynb`.

## Purpose and functionality of `test_generate.py`

This python script generates random network properties and test network generation functions for visual inspection.

- This `create_random_properties` test function generates random values for pore diameters (1.0 - 10.0), throat diameters (0.5 - 6.0), and coordination numbers (0 - 8, converted to integers).
- This  `test_generate_network` test function takes random properties and network parameters (number of nodes, center channel) to generate an OpenPNM network using the `generate_network` function and then prints the generated network to the console for visual inspection.
- This `test_generate_network2` function is similar to `test_generate_network`. This function additionally takes probability density functions (pdfs) for each property and allows specifying an average coordination number.

## Purpose and functionality of `test_metrics.py`

This python script demonstrates generating a test image with simple ellipses, applying the diameter extraction functions, and optionally displaying the output for a given sample. It also includes advanced tests with generated cylindrical 3D structures to show how the functions can handle more complex image data.

- This `test_feret_diameter` function processes an image to calculate and return the feret diameters, which describe the extent of a projected area of the image.
- The `test_extract_diameters` and `test_extract_diameters2` functions extract pore and throat diameters from images. They showcase different methodologies, with `test_extract_diameters2` using a method that doesn't rely on `PoreSpy's` built-in SNOW algorithm. For more complex image analysis tasks, we will need to explore advanced techniques like watershed segmentation or ML-based segmentation methods.

## Purpose and functionality of `test_train_pinn.py`

This python script is designed to validate the functionality of the `train_PINN` function, which trains a PINN for flow on a micromodel. The tests are built using the `unittest` framework. It ensures that the `train_PINN` function behaves correctly under various scenarios, including typical use cases, edge cases, and error scenarios. The test can be executed by running the script directly `python test_train_pinn.py`, which invokes `unittest.main()` (entry point for running the unit tests). This will run all the defined tests within the `test_train_pinn.py` and provide a terminal report on their success or failure.

- **`setUp()`**: Initializes test parameters such as neural network weights, biases, optimizer, loss function, collocation points, and boundary conditions. This method runs before each test case.

- **`test_train_PINN_output()`**:
    - **Purpose**: Tests the output of `train_PINN` function to ensure that valid results (best parameters, best loss, all losses, all epochs) are returned after PINNs training.
    - **Assertions**: Checks if the results are not `None` and if the lengths of the loss and epoch lists match the number of epochs.

- **`test_train_PINN_type()`**:
    - **Purpose**: Verifies the type of each output from `train_PINN` function.
    - **Assertions**: Ensures that the outputs (trainable parameters, loss, losses, epochs) have the expected data types.

- **`test_train_PINN_shape()`**:
    - **Purpose**: Checks if the shape (length) of the returned parameters and loss/epoch lists match the expected values.
    - **Assertions**: Confirms that the number of returned parameters matches the input, and the loss and epoch lists have the correct length.

- **`test_train_PINN_values()`**:
    - **Purpose**: Ensures that the best loss returned by `train_PINN` function is less than or equal to the initial loss from the loss function.
    - **Assertions**: Verifies that the best loss is reasonable by comparing it with the initial loss.

- **`test_train_PINN_exceptions()`**:
    - **Purpose**: Tests that `train_PINN` raises appropriate exceptions for invalid inputs.
    - **Assertions**: Checks that `TypeError` is raised for invalid parameter types and that `ValueError` is raised for invalid epoch values (e.g., negative epochs).

- **`test_train_PINN_edge_cases()`**:
    - **Purpose**: Tests edge cases for the `train_PINN` function, such as training with zero or negative epochs.
    - **Assertions**: Ensures that `ValueError` is raised when invalid epoch values are passed, and verifies that valid results are returned for small and large epoch values.

- **`test_train_PINN_different_optimizers()`**:
    - **Purpose**: Tests `train_PINN` function with different `optax` optimizers (e.g., SGD, Adam, RMSProp).
    - **Assertions**: Ensures that valid parameters and losses are returned for each optimizer.

- **`test_train_PINN_different_loss_functions()`**:
    - **Purpose**: Validates that `train_PINN` function works correctly with different loss functions.
    - **Assertions**: Checks that valid parameters and finite loss values are returned for each loss function.

- **`test_train_PINN_multiple_layers_nodes()`**:
    - **Purpose**: Tests `train_PINN` function with different numbers of hidden layers and nodes.
    - **Assertions**: Ensures that valid parameters and losses are returned for each combination of hidden layers and nodes.