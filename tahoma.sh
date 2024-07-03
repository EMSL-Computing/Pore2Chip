#!/bin/bash

# This script automates cloning, building (if necessary), and installing the Pore2Chip python package.

# Clone the Pore2Chip repository from GitHub
git clone https://github.com/aramyxt/Pore2Chip.git

# Change directory to the cloned repository
cd Pore2Chip

# Try building the package (assumes a setup.py is present)
python3 -m build || true

# Install the package from the local dist directory
python3 -m pip install pore2chip --no-index --find-links dist/