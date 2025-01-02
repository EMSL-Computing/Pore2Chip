
.. Installation
.. =======================

.. The OpenPNM and PoreSpy libraries are required to analyze XCT images. PoreSpy is used to generate a 
.. pore network that is used to extract pore size distribution, pore throat size distribution, and pore coordination numbers. 
.. OpenPNM is used to construct a new 2D pore network that will be used to create the micromodel design.

.. Install using PiP:

.. .. code-block:: console
        .. $ pip install pore2chip

Installation
============

The OpenPNM and PoreSpy libraries are required to analyze XCT images. PoreSpy is used to generate a 
pore network that is used to extract pore size distribution, pore throat size distribution, and pore coordination numbers. 
OpenPNM is used to construct a new 2D pore network that will be used to create the micromodel design.

Install using PiP:

.. code-block:: console

   $ pip install pore2chip

Install from source:

.. code-block:: console

   $ git clone https://github.com/EMSL-Computing/Pore2Chip.git 
   $ cd Pore2Chip
   $ python3 -m build
   $ python3 -m pip install pore2chip --no-index --find-links dist/

Creating a Conda environment:

.. code-block:: console

   $ conda create -n pore2chip python=3.9
   $ conda activate pore2chip
   $ pip install pore2chip

Building a Docker Image with Jupyter Notebook:

.. code-block:: console

   $ git clone https://github.com/EMSL-Computing/Pore2Chip.git 
   $ cd Pore2Chip
   $ docker build -t pore2chip
   $ docker run -p 8888:8888 pore2chip