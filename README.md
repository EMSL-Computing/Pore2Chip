# Pore2Chip: All-in-One Python Tool for Soil Microstructure Analysis and Micromodel Design

## What is Pore2Chip?
Pore2Chip is a Python module designed to streamline the process of analyzing X-ray computed tomography (XCT) images of soil and creating 2D micromodel designs based on that analysis. It leverages the power of open-source libraries like OpenPNM, PoreSpy, and drawsvg to  extract key information about the soil's porous structure and translate it into a blueprint for microfluidic simulations or physical "lab-on-a-chip" devices developed using additive manufacturing.

### A workflow for model-data-experiment (ModEx) design:

Below is a conceptual figure, workflow, and vision for this all-in-one Python tool. The working principle starts with XCT imaging files, which will be characterized for soil structure-property relationships and then transformed into a 2D rendering applicable to pore-scale micromodel building. Micromodel experiments will then be used with PFLOTRAN/OpenFOAM/PINNs process models to simulate flow and reactive transport for calibration and V&V.

**ModEx for SoilChip**: Lab-on-a-chip designs to accelerate ModEx workflows informed by soil datasets. Key Steps in the ModEx loop include:

(1) **XCT Imaging of Soil Core and Aggregates (Pore2Chip):** High-resolution X-ray computed tomography (XCT) imaging captures detailed 3D structures, providing a foundational understanding of the physical characteristics.

(2) **3D Pore Network Characterization (Pore2Chip):** The 3D pore network is analyzed to determine pore size frequency and distribution, which is critical for understanding flow and transport properties.

(3) **Transform 3D Pore Network into 2D Rendering (Pore2Chip):** The complex 3D network is simplified into a 2D rendering for easier analysis and visualization.

(4) **Build Micromodels for Environmental Experiments (Pore2Chip):** Micromodels replicate environmental conditions, enabling controlled experiments to observe fluid flow and chemical species degradation.

(5) **Microscale Experimental Data on Chemical Hotspots (Chip2Flow):** Detailed experiments using techniques like ToF-SIMS and SEM-EDS provide data on chemical hotspots within the porous media.

(6a) **Pore-Scale Multi-Physics Modeling (Chip2Flow):** Simulations model fluid flow, heat transfer, and chemical reactions at the pore scale, which is needed to predict system behavior under different environmental conditions.

(6b) **Calibration and Validation (Chip2Flow):** Predictive AI/ML-enabled models are calibrated and validated using experimental data for accuracy and reliability.

(7a) **Understanding Fluid Flow and Species Degradation in Soil Core Experiments (Chip2Flow):** Experiments on soil cores provide vital information on fluid flow and chemical species degradation, connecting back to micromodel generation.

(7b) **Upscaled Properties (Chip2Flow):** Properties and behaviors observed at smaller scales are upscaled to larger scales (mm to cm) for real-world application.

**Conclusion:** The iterative ModEx loop continuously improves multi-physics process models through integration with experimental data, leading to more accurate predictions for system evolution and rhizosphere function applications. Additional specifics on extending this to Critical Minerals and Material (CMM) science applications is here [Download the SoilTwin poster pdf](https://github.com/EMSL-Computing/Pore2Chip/blob/main/paper/material/Pore2Chip_Chip2Flow_Specifics.pdf)

![ModExLoop](https://github.com/EMSL-Computing/Pore2Chip/blob/main/example_outputs/ModEx_Loop_SoilChip.jpg)

## Capability summary: What the Pore2Chip module can do?
* Extract pore sizes and pore throat sizes
* Extract pore connectivity numbers
* Get miscellaneous pore information such as feret diameters
* Generate a micromodel design that is representative of input XCT soil data
* Export the design as an SVG file
* Export design as a DXF file (see DXF section)

### Unveiling the Hidden World of Soil: Pore Structure Analysis

Pore2Chip empowers you to delve into the intricate details of soil microstructure by:

* **Quantifying Pore Sizes and Throats:**  It precisely measures the size distribution of pores and pore throats within the soil sample. This information is crucial for understanding fluid flow properties and transport phenomena within the soil.
* **Mapping Pore Connectivity:**  Pore2Chip calculates the number of connections between pores, providing valuable insights into how fluids can move through the soil network.
* **Extracting Diverse Pore Metrics:**  In addition to size and connectivity, Pore2Chip can extract various other pore characteristics, such as feret diameters (the greatest distance a pore can span in a specific direction).

### Bridging the Gap: From Soil Data to Microfluidic Designs

Pore2Chip goes beyond analysis by translating the extracted data into actionable outputs:

* **Micromodel Design Generation:**  Based on the characterization of the soil's pore network, Pore2Chip generates a 2D blueprint that closely resembles the actual pore structure. This design serves as a foundation for microfluidic simulations or the fabrication of physical micromodels using Photolithography or Laser Etching.
* **SVG File Export:**  The micromodel design is exported in a versatile SVG (Scalable Vector Graphics) format, ensuring compatibility with various simulation software and design tools.
* **DXF Export (Optional):**  For users working with computer-aided design (CAD) programs, Pore2Chip can optionally export the design in DXF (Drawing Exchange Format)  facilitating integration into CAD workflows (Note:  DXF export functionality may require additional configuration).

In essence, Pore2Chip offers a comprehensive solution for researchers and engineers working with soil microstructures. It efficiently bridges the gap between XCT data and micromodel development, paving the way for a deeper understanding of soil behavior and the creation of advanced microfluidic devices for diverse applications.

## Getting Started
Here is a link to the documentation hosted on Github Pages: https://emsl-computing.github.io/Pore2Chip/

The OpenPNM and PoreSpy libraries are required to analyze XCT images. PoreSpy is used to generate a pore network that is used to extract pore size distribution, pore throat size distribution, and pore coordination numbers. OpenPNM is used to construct a new 2D pore network that will be used to create the micromodel design.

Example input images can be found in the "bean_bucket_100" folder. Full dataset can be found here: https://github.com/EMSL-MONet/CommSciMtg_Nov23/

Install using PiP:
```
pip install pore2chip
```
Install from source:
```
git clone https://github.com/EMSL-Computing/Pore2Chip.git 
cd Pore2Chip
python3 -m build
python3 -m pip install pore2chip --no-index --find-links dist/
```
...or alternatively:
```
git clone https://github.com/EMSL-Computing/Pore2Chip.git 
python3 pip install -e ./Pore2Chip
```

Creating a Conda environment:
```
conda create -n pore2chip python=3.9
conda activate pore2chip
pip install pore2chip
```

Building a Docker Image with Jupyter Notebook:
```
git clone https://github.com/EMSL-Computing/Pore2Chip.git 
cd Pore2Chip
docker build -t pore2chip
docker run -p 8888:8888 pore2chip
```
This should output URLs that you can copy and paste into a browser so that you can access the Jupyter Notebook server.

## Example Usage
In the following examples, pore and throat diameters as well as coordination numbers are hard coded values. These values can be extracted from XCT images using the ```metrics``` library.
```
from pore2chip import generate, export

# Shape of the micromodel (number of pores n x n).
n = 5
# Random values for pore, throat diameters and coordination numbers. Can be any length.
arr_pore = [4.0, 9.0, 4.5, 8.4, 14.0, 7.6, 5.0]
arr_throat = [7.0, 5.5, 3.5, 1.4, 5.8, 4.3, 8.8, 8.4, 4.0]
arr_coord = list(range(0, 4))

network = generate.generate_network(n, n, arr_pore, arr_throat, arr_coord)

design = export.network2svg(network, n, n, 100, 100)

design.save_svg('network.svg')
```
![output](https://github.com/EMSL-Computing/Pore2Chip/blob/main/example_outputs/network_from_values.svg)

This package can also generate images without throats and simulate "pores" as "grain" particulates.
```
from pore2chip import generate, export

# Shape of the micromodel (number of "grains" n x n).
n = 5
# Random values for pore, throat diameters and coordination numbers. Can be any length.
arr_grain = [4.0, 3.0, 4.5, 2.4, 14.0, 7.6, 5.0]

network = generate.generate_network(n, n, arr_grain, None, None)

design = export.network2svg(network, n, n, 100, 100)

design.save_svg('grain_network.svg')
```
![output2](https://github.com/EMSL-Computing/Pore2Chip/blob/main/example_outputs/grain_network.svg)

## Converting to PNG
There are many ways to convert an SVG image to a rasterized image format using only Python, such as rendering it using```CairoSVG```.
The recommended way to do this is to use ```svglib``` and ```reportlab```. NOTE: ```reportlab>=4.0.0``` requires ```pycairo```, and by extension, the ```cairo``` library, which cannot be installed by PiP by itself. If you are using Windows, it is recommended to install ```reportlab=3.6.13```.
```
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

rldrawing = svg2rlg('network.svg')
renderPM.drawToFile(rldrawing, 'network.png', fmt='PNG')
```

## Micromodel design -- Getting an STL from the SVG for additive manufacturing 
This is a generalized workflow for getting an SVG to an STL file:

1. Generate the svg using Pore2Chip
2. Use a vector image program such as Inkscape to combine all paths into one path. In Inkscape, this would be ```Paths -> Union```
3. Export the new image as a .DXF file
4. Import the DXF file into the CAD software of your choosing, such as FreeCAD
5. Extrude the shape of the pores and use it as a negative to create the micromodel
6. Export to STL

Example result in Solidworks:
![solidworks_ex](https://github.com/EMSL-Computing/Pore2Chip/blob/main/example_outputs/cad_mockup2.PNG)

There are many other methods to print the micromodel design onto physical materials.
Another example of fabricating lab-on-chip micromodels is using the design to etch it on a surface. Example:

![fabricated](https://github.com/EMSL-Computing/Pore2Chip/blob/main/example_outputs/fabricated_chip.jpg)

This laser etching functionality is available at EMSL. Please contact us for more information as you develop these micromodels for fabrication.

It is highly recommended to try the Python library ```svglob``` to combine SVG paths without using an external program:
https://github.com/deckar01/svglob/tree/master

Alternatively, ```stl_tools``` can be used to turn a rasterized image into an STL:
https://github.com/thearn/stl_tools

## DXF Exporting
While the ```export``` module can export a DXF file, it can only create pores as circles. It is recommended to export the micromodel as an SVG file and make desired adjustments to it. 
This way, you have more control over the shape and can then convert the SVG to a DXF file.

## VTK Exporting

Download ParaView software for VTK file visualization: [Viz software download](https://www.paraview.org/download/). The dataset for visualizing Pore2Chip VTK files is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11211760.svg)](https://doi.org/10.5281/zenodo.11211760). For more information related to the Pore2Chip VTK file outputs, see ```src/pore2chip/io.py```. Also, please refer to the following links for additional details on file formats:
* [VTK File Formats Documentation](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html) or
* [VTK File Format User Guide](https://www.princeton.edu/~efeibush/viscourse/vtk.pdf) 

## For EMSL Tahoma Users
To use the library with Tahoma Open OnDemand:
1. Start a Jupyter Notebook instance in the EMSL OnDemand dashboard
2. Create a Python virtual environment in the terminal
3. Install Pore2Chip in the virtual environment
4. In Jupyter Notebook, set your kernel to your python environment

For more information, see [user guide for EMSL Open OnDemand](https://www.emsl.pnnl.gov/MSC/UserGuide/ood/overview.html)

## Known Issues
* The coordination algorithm may create pores of coordination of 1, despite the given coordination list not having 1's. This is being worked on.

## Work-in-progress
* The ability to import and export pore networks in CSV or VTK file formats needed for multi-physics process modeling (e.g., using PFLOTRAN)
* Physics-informed machine learning needed for flow, thermal, and reactive-transport modeling (e.g., advanced physics-informed neural networks, operator learning methods)
* Meshfiles needed for CFD modeling (e.g., using OpenFOAM)
* GUI for the Docker container

### Within the Pore2Chip platform

Our team is developing advanced modeling techniques to simulate complex multi-physics phenomena numerically. These numerical simulations are essential for understanding and predicting fluid flow behavior, chemical species degradation, and nutrient intake within the micromodels for reactive-transport applications.

The micromodels generated by Pore2Chip can be meshed for multi-physics simulations to understand flow and reactive-transport modeling better. Here, we show a structured mesh capability (see [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11211760.svg)](https://doi.org/10.5281/zenodo.11211760)) that can be generated for PFLOTRAN-based multi-physics modeling. The mesh is a grid of points to solve fluid flow and reactive transport within the lab-on-chip.

### Key modeling capabilities that we are developing include

**Physics-Informed Neural Networks:** Deep neural networks integrate physical laws with data, enhancing machine learning models' predictive accuracy and reliability.

**PFLOTRAN Input Decks:** PFLOTRAN (see [PFLOTRAN Documentation](https://www.pflotran.org/))  is a massively parallel subsurface flow and reactive transport code. It simulates the movement of fluids and the transport of chemical species in this porous micromodel.

**CFD Simulations:** Computational Fluid Dynamics (CFD) simulations are utilized to simulate fluid flow within the micromodel, providing insights into the dynamics of fluid movement and interaction with the solid matrix.

## Example Jupyter notebooks (basic and advanced usage):
* Example-1: Micromodel Creation from 50 x 50 XCT Data
* Example-2: Micromodel Creation from 100 x 100 XCT Data
* Example-3: Micromodel Pore Stats Using PoreSpy
* Example-4: Flow and transport simulations on micromodels using PoresPy
* Example-5: VTK exports for visualization of micromodel in Paraview
* Example-6: Finite Difference Method (FDM) Numerical Simulation of 2D Steady-State Flow on XCT images
* Example-7: 2D Steady-State Flow on XCT image with Physics-informed Neural Network (PINN)

## Video recordings:
Below are some video links explaining Pore2Chip functionalities:
* [EMSL LEARN Webinar Series](https://www.emsl.pnnl.gov/events/emsl-learn-webinar-series-pore2chip-python-tool-terraforms)
* [Video on YouTube](https://youtu.be/_DTLtYBAZSo?si=vMsfUm3w5hLyzNBr) 

## Authors
* Aramy Truong (lead developer), EMSL (<aramy.truong@pnnl.gov>)
* Maruti Mudunuru (co-lead developer), PNNL (<maruti@pnnl.gov>)
* Erin Rooney, USDA (<erin.rooney@usda.gov>)
* Arunima Bhattacharjee, EMSL (<arunimab@pnnl.gov>)
* Tamas Varga, EMSL (<tamas.varga@pnnl.gov>)
* Lal Mamud (developer), PNNL (<lal.mamud@pnnl.gov>)
* Xiaoliang (Bryan) He, PNNL (<xiaoliang.he@pnnl.gov>)
* Anil Krishna Battu, EMSL (<anilkrishna.battu@pnnl.gov>)
* Satish Karra, EMSL (<karra@pnnl.gov>)

## Development and questions
We welcome your contributions to Pore2Chip! This includes bug reports, bug fixes, improvements to the documentation, feature enhancements, and new ideas. 

**Copyright Guidelines:**

To ensure the project's overall licensing remains compatible, please keep the following in mind:

* **Datasets:** Avoid including datasets with restricted licenses that don't allow free use or modification. These can create conflicts with the project's license.
* **Code snippets:** Avoid using code snippets with restricted licenses that don't allow free use or modification.

**Contributing to Pore2Chip:**

We appreciate all contributions, big or small! Here's how to get involved:

* **Fork the repository and create a pull request:** This is the preferred method for contributing code changes.
* **Formatting the code using yapf:**
 ```
 yapf -i --style=pep8 <my_code>.py
 ```
 
 or if you edit multiple files, you can do the following:

```
yapf -i --style=pep8 --recursive .
```

* **Contact Aramy Truong (<aramy.truong@pnnl.gov>) and/or Maruti Mudunuru (<maruti@pnnl.gov>):** If you have questions or need help getting started.

Additionally, your contributions can be as simple as:

* Fixing typos
* Implementing a new feature calculator
* Developing a novel feature selection process

**No matter your skill level, your help is valuable!**

### Directory structure for contribution

 ```
 tree -L 2 . >> tree.txt
 ```

```
.
├── Dockerfile
├── LICENSE.md
├── README.Docker.md
├── README.md
├── compose.yaml
├── dist
│             ├── pore2chip-0.0.8-py3-none-any.whl
│             └── pore2chip-0.0.8.tar.gz
├── docker_requirements.txt
├── example_outputs
│             ├── ModEx_Loop_SoilChip.jpg
│             ├── cad_mockup.PNG
│             ├── cad_mockup2.PNG
│             ├── flow
│             ├── grain_network.png
│             ├── grain_network.svg
│             ├── micromodel.npy
│             ├── network.dxf
│             ├── network.png
│             ├── network.svg
│             ├── network1.png
│             ├── network1.svg
│             ├── network2.png
│             ├── network2.svg
│             └── network_from_values.svg
├── examples
│             ├── bean_bucket_100
│             ├── example_1_50x50_to_model.ipynb
│             ├── example_2_100x100_to_model.ipynb
│             ├── example_3_model_properties.ipynb
│             ├── example_4_porespy_analysis.ipynb
│             ├── example_5_vtk_exporting.ipynb
│             ├── example_6_flow_2d_numerical_on_XCT.ipynb
│             └── example_7_flow_2d_pinn_on_XCT.ipynb
├── paper.bib
├── paper.md
├── pyproject.toml
├── requirements.txt
├── src
│             ├── flow
│             ├── pore2chip
│             └── pore2chip.egg-info
├── tahoma.sh
├── tests
│             ├── test_coordination.py
│             ├── test_export.py
│             ├── test_filter_im.py
│             ├── test_generate.py
│             └── test_metrics.py
└── tree.txt

11 directories, 40 files
```

## Acknowledgements
This research was performed on a project award (Award DOIs: 10.46936/ltds.proj.2024.61069/60012423; 10.46936/intm.proj.2023.60674/60008777; 10.46936/intm.proj.2023.60904/60008965) from the Environmental Molecular Sciences Laboratory, a DOE Office of Science User Facility sponsored by the Biological and Environmental Research program under contract no. DE-AC05-76RL01830. The authors acknowledge the contributions of Michael Perkins at PNNL’s Creative Services, who developed the conceptual graphics in this paper.

PNNL-SA-197910

## Disclaimer
This research work was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
