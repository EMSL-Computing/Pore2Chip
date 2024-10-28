---
title: 'Pore2Chip: All-in-one python tool for soil microstructure analysis and micromodel design'
tags:
  - Python
  - Micromodel design
  - Model-data-experiment
  - Visualization
  - Porous media
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
 - name: Environmental Molecular Sciences Laboratory, Pacific Northwest National Laboratory, Richland, WA, USA
   index: 1
 - name: Pacific Northwest National Laboratory, Richland, WA, USA
   index: 2
 - name: USDA-NRCS National Soil Survey Center, Lincoln, NE, USA
   index: 3
date: 27 October 2024
bibliography: paper.bib

---

# Summary

The `Pore2Chip` is a Python package designed to create two-dimensional micromodels using extracted data from three-dimensional X-ray computed tomography (XCT) images. 
Micromodels are two-dimensional representations of pore systems that replicate natural porous media, enabling detailed study of fluid dynamics, reactive-transport, and thermal energy at the pore scale [@karadimitriou2012review; @anbari2018microfluidic]. 
These models are crucial for advancing soil science by providing insights into microbial interactions and chemical processes in Earth and energy systems [@isah2022fluid; @zhu2022microfluidics; @@aralekallu2023development].
This `Pore2Chip` package helps analyze soil structure and function, allowing for the investigation of environmentally significant biogeochemical processes that impact soil organic matter (SOM) decomposition and loss, oxygen concentrations, and nutrient availability in disturbed or managed soils.
It can produce pore networks that accurately represent soils, convert any 3D pore network into scalable SVG images with precise pore and pore throat sizes, and conduct computational simulations on these images. 
The extracted data includes characterizations of the pore network using major water retention and flow metrics relevant to porous materials. 
Key metrics encompass pore size distributions, pore throat size distributions, porosity, tortuosity, and connectivity (pore coordination numbers).
The software's final output is a 2D lab-on-chip micromodel designs representing a soil core or its aggregate. It can be printed using additive manufacturing methods such as laser etching, 3D printing, and photolithography. 
These 3D-printed designs accurately depict the XCT-resolved pore networks for soil science applications.

# Statement of need

The resilience of agricultural and natural landscapes is intrinsically connected to soil structure. 
Land management (e.g., tillage, grazing, and fire) and associated impacts (e.g., compaction and pore-clogging) along with climate disturbances (e.g., freeze-thaw, flooding, and sea level rise) can transform soil microstructure [@liu2018; @feng2020; @stoof2016; @oliveira2022; @rooney2022]. 
These changes in the soil microstructure determine the flow of water, solutes, and gases, as well as SOM retention, transport, and distribution [@hamamoto2010; @waring2020; @guo2024; @bailey2017]. 
Simplified, homogeneous pore networks provide innovative demonstrations of how water, solutes, and microbes interact [@bhattacharjee2022] but need more accurate representations of soil properties. 
Heterogeneous synthetic habitats are more realistic but time-consuming to design and do not include pore network characteristics, such as pore connectivity or pore throat measurements. 
Incorporating pore dynamics into soil models, such as SOM degradation, enables dynamic predictions for soil responses under changing pore networks [@davidson2011; @moyano2018].

Our understanding of how pore network characteristics govern hydrobiogeochemical processes in pre- and post-disturbance soils and across soil types is constrained by current micromodel-based experimental designs. Currently, there is no software available to seamlessly provide various micromodel designs that researchers can test and validate with minimal computational cost [@dentz2023; @oostrom2014]. 
`Pore2Chip` allows us to overcome this barrier by providing the intended users, such as earth scientists and lab-on-chip instrument specialists, with easy-to-use research software for lab-on-chip designs. 
Specifically, the `Pore2Chip`-based data worth analysis of high-resolution XCT images allows us to fill this experimental design gap by providing a 3D model of the microscale pore structure. 
This `Pore2Chip` capability allows the users to build a representative quasi-2D pore network along with first-order, fast, and reasonably accurate flow models that can be linked with soil experiments. 
These `Pore2Chip` flow models are built using recent advances in physics-informed neural networks [@new2024], laying the foundation to accelerate numerical simulations and improve the fidelity of predictions in microscale environments. 
Moreover, `Pore2Chip` allows one to assess the impact of various system parameters, such as pore structures, fluid properties, and flow conditions, needed to develop optimal micromodel experiments. 
Such a capability can guide model-experiment-data (ModEx) integration at the microscale, allowing for upscaling microscale processes and predictions of dynamic soil properties and functions. 
See \autoref{fig:fig1}.

![A high-level overview of essential steps in Pore2Chip-based micromodel designs informed by soil dataset. The iterative ModEx loop continuously improves multi-physics process models by integrating experimental data, leading to more accurate predictions for soil carbon cycling and rhizosphere function applications.\label{fig:fig1}](figures/2_ModEx_Loop_SoilChip.png)

