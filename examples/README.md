# Description of the files in the `examples` folder

This folder provides sample XCT image dataset for running python scripts and Jupyter notebooks. You can launch Jupyter Notebooks or open the desired Jupyter Notebook file and execute the cells to run the code and view the results.

## Data folder

- **bean_bucket_100/**: This folder contains related XCT image datasets required by the notebooks.

## Jupyter notebooks using Pore2Chip capabilities

- **example_1_50x50_to_model.ipynb**: This notebook demonstrates how to create a micromodel from start to finish using a part of the XCT image data from a soil core (50x50).

- **example_2_100x100_to_model.ipynb**: This notebook demonstrates how to create a micromodel from start to finish using a part of the XCT image data from a soil core (100x100) dataset.

- **example_3_model_properties.ipynb**: In this notebook, various micromodel and XCT data properties are analyzed and visualized. This includes calculating porosity and finding feret diameters

- **example_4_porespy_analysis.ipynb**: This notebook includes analysis techniques using the PoreSpy library on the generated example micromodel.

- **example_5_vtk_exporting.ipynb**: This notebook provides a guide on exporting micromodels and data to VTK format needed for flow simulations.

- **example_6_flow_2d_numerical_on_XCT.ipynb**: This notebook shows a 2D numerical flow analysis on the generated micromodel.

- **example_7_flow_2d_pinn_on_XCT.ipynb**: This notebook explores 2D flow analysis using Physics-Informed Neural Networks on the generated micromodel using XCT data.