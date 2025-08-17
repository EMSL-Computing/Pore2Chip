---
title: 'Pore2Chip: All-in-one python tool for soil microstructure analysis and micromodel design'
tags:
  - Python
  - Micromodel design
  - Model-data-experiment
  - Visualization
  - porous-media
  - flow
  - data-analysis
authors:
  - name: Aramy Truong
    corresponding: true
    orcid: 0009-0002-7467-7957
    affiliation: "1"
  - name: Maruti K. Mudunuru
    corresponding: true
    orcid: 0000-0002-2158-7723
    affiliation: "2"
  - name: Erin C. Rooney
    orcid: 0000-0002-9121-7699
    affiliation: "3"
  - name: Arunima Bhattacharjee
    orcid: 0000-0001-7482-8722
    affiliation: "1"
  - name: Tamas Varga
    orcid: 0000-0002-5492-866X
    affiliation: "1"
  - name: Md Lal Mamud
    orcid: 0000-0002-5764-1058
    affiliation: "2"
  - name: Xiaoliang He
    orcid: 0000-0003-3835-1917
    affiliation: "2"
  - name: Anil K. Battu
    orcid: 0000-0001-7724-5100
    affiliation: "1"
  - name: Satish Karra
    orcid: 0000-0001-7847-6293
    affiliation: "1"
affiliations:
 - name: Environmental Molecular Sciences Laboratory (EMSL), Pacific Northwest National Laboratory, Richland, WA, USA
   index: 1
 - name: Pacific Northwest National Laboratory (PNNL), Richland, WA, USA
   index: 2
 - name: USDA-NRCS National Soil Survey Center, Lincoln, NE, USA
   index: 3
date: 27 October 2024
bibliography: paper.bib

---

# Summary

The `Pore2Chip` Python package is designed to create 2D micromodels using extracted data from 3D X-ray computed tomography 
(XCT) images. 
This package helps analyze soil structure and function, allowing for the investigation of hydro-biogeochemical processes 
that impact mineral extraction and reactivity, oxygen concentrations, and nutrient availability in disturbed or managed soils. 
Key metrics encompass pore size distributions, pore throat size distributions, and connectivity (pore coordination numbers). 
The final output is a 2D scalable SVG design representing a core or aggregate. 
Designs can be fabricated with methods such as laser etching, 3D printing, and photolithography.

# Statement of need

The resilience of agricultural and natural landscapes is intrinsically connected to soil structure. 
Land management (e.g., tillage, grazing, and fire) and associated impacts (e.g., compaction, pore-clogging) can transform soil microstructure [@Stoof2016; @Liu2018; @Feng2020; @deOliveira2022; @Rooney2022]. 
These changes in the soil microstructure determine the flow of water, solutes, and gasses as well as mineral retention, transport, and distribution [@hamamoto2010excluded; @bailey2017differences; @Waring2020]. 
Simplified, homogeneous pore networks provide innovative demonstrations of how water, solutes, and microbes interact [@Bhattacharjee2022] but need more accurate representations of soil properties. 
Creating realistic heterogeneous habitats is time-consuming and does not include pore network characteristics, such as pore 
connectivity. 
Incorporating pore dynamics into soil models such as chemical species degradation enables dynamic predictions for soil 
responses under changing pore networks [@davidson2012d; @moyano2018diffusion].

The need for software that can generate various micromodel designs that researchers can test and validate with minimal computational cost [@Dentz2023; @Oostrom2014] is increasing.
`Pore2Chip` allows this functionality by providing the intended users, such as earth scientists and lab-on-chip instrument specialists, with easy-to-use research software for lab-on-chip designs. 
Specifically, the Pore2Chip-based information analysis of XCT images allows researchers to fill this experimental design gap by enabling the ability to build a representative quasi-2D pore network along with first-order, fast, and reasonably accurate flow models that can be linked with experiments. 
These flow models are built using recent advances in physics-informed neural networks [@New2024], laying the foundation to accelerate numerical simulations and improve the fidelity of predictions in microscale environments. 
Moreover, `Pore2Chip` allows one to assess the impact of various system parameters, such as pore structures, fluid properties, and flow conditions, needed to develop optimal micromodel experiments. 
Such a capability can guide model-experiment-data (ModEx) integration at the microscale, allowing for upscaling microscale processes and predictions of dynamic soil properties and functions (see \autoref{fig:fig1}).

## Main features and differences with other tools

`Pore2Chip` addresses complex pore structures by representing pore networks as connected shapes, unlike older sphere packing algorithms. 
This enables users to easily create and control pore networks representing various real-world conditions. `Pore2Chip` offers experimental design capabilities that 
cannot be achieved by existing software such as epyc [@Dobson2022]. 
`Pore2Chip` provides support and reproducibility for developing lab-on-chip experimental designs uniformly across different soil datasets with fast, reasonably 
accurate, first-order flow modeling capabilities. Microscale experiments using `Pore2Chip` micromodels may target both abiotic and biotic 
processes and be integrated into modeling efforts such as water flow modeling, reactive transport modeling, and microbial activity simulations. 


## Implementation details and support libraries

Using `Porespy` [@Gostick2016], `OpenPNM` [@Gostick2016] and various graphics rendering libraries (e.g., drawsvg, ezdxf, svglib, cairosvg, reportlab), 
`Pore2Chip` renders SVG or DXF micromodel designs of the generated network. Output designs are scalable and adjustable based 
on the target porosity of the micromodel. It can also be exported as micromodel data in VTK formats for 
visualization in Paraview or microfluidic simulations with open-source software such as `PFLOTRAN` (https://www.pflotran.org), `OpenFOAM` (https://www.openfoam.com), 
and other physics-informed neural network modules. If the user wants to extract data from XCT images, `Pore2Chip` has image filtering and network extraction 
modules utilizing Otsu thresholding and `PoreSpy`. Though, generation function can also work with data extracted by other software as long as it is an array of values that 
Python can read. 

\autoref{fig:fig2} provides a high-level overview of the repository structure and example use cases (\autoref{fig:fig1}) within the `Pore2Chip` repository. 

# Figures

![A high-level overview of essential steps in Pore2Chip-based micromodel designs informed by soil dataset. The iterative ModEx loop continuously improves multi-physics process models by integrating experimental data, leading to more accurate predictions for soil carbon cycling and rhizosphere function applications.\label{fig:fig1}](figures/2_ModEx_Loop_SoilChip.jpg)

![An overview of the Pore2Chip repository structure, detailed example notebooks, and built distributions.\label{fig:fig2}](figures/3_Workflow.png)

# Acknowledgement and disclaimer

This research was performed with support from the Environmental Molecular Sciences Laboratory, a DOE Office of Science User Facility sponsored by the Biological and Environmental Research program under contract no. DE-AC05-76RL01830. 
The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof. 

# References
