import sys
import os
import time
import numpy as np
import drawsvg as dr
import math
import ezdxf
#import cairosvg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from pathlib import Path

# Set up the module path (I am in tests folder)
# Modify these lines if the pore2chip package is not in the PYTHONPATH
mod_path = Path("__file__").resolve().parents[2]
sys.path.append(os.path.abspath(mod_path))

# Importing specific functions for exporting network data to SVG and DXF formats
import pore2chip
#from pore2chip.src.pore2chip.export import network2svg, network2dxf
from pore2chip.export import network2svg, network2dxf


def generate_network():
    """
    Generate a simple network with 3x3 pores and straight throats.
    
    Returns:
        dict: A dictionary representing the network structure, with coordinates, diameters of pores,
              connections (throats), and their diameters with the following keys:
                - 'pore.coords': List of pore coordinates (x, y).
                - 'pore.diameter': List of pore diameters.
                - 'throat.conns': List of pore indices connected by throats.
                - 'throat.diameter': List of throat diameters.
    """
    return {
        "pore.coords": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2),
                        (1, 2), (2, 2)],
        "pore.diameter":
        [1 for _ in range(9)],  # Uniform diameter for all pores
        "throat.conns": [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
                         (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8)],
        "throat.diameter":
        [0.5 for _ in range(12)]  # Uniform diameter for all throats
    }


def test_network2svg(network):
    """
    Test network2svg function with pore_shape='circle' and throat_random=0 (straight throats)
    by generating an SVG file with specified parameters.
    
    Args:
        network (dict): The network structure from generate_network.
    """

    # Generate an SVG from the network with specified visual parameters
    svg_result = network2svg(
        network,
        n1=3,  # grid size in x-direction
        n2=3,  # grid size in y-direction
        d1=300,  # SVG image size in x-direction
        d2=300,  # SVG image size in y-direction
        pore_shape='circle',  # visual shape of pores
        throat_random=0)  # throat connection randomness

    # Save the SVG to a file
    svg_result.save_svg('grain_network.svg')

    # Convert the SVG to PNG using cairosvg for additional usability
    #cairosvg.svg2png(url="grain_network.svg", write_to="grain_network.png")

    # Convert the SVG to PNG using renderlab for additional usability
    rldrawing = svg2rlg('grain_network.svg')
    renderPM.drawToFile(rldrawing, 'grain_network.png', fmt='PNG')


def test_network2dxf(network):
    """
    Test network2dxf function with throat_random=0 (straight throats)
    by generating a DXF file with specified parameters.
    
    Args:
        network (dict): The network structure.
    """

    # Generate a DXF from the network with straight throats
    dxf_result = network2dxf(network, throat_random=0)

    # Save the DXF to a file
    dxf_result.saveas("test_dxf_1.dxf")


def main():
    """
    Main function to orchestrate network generation and testing of SVG and DXF export functions.
    """

    # Generate network
    network = generate_network()

    # Test exporting the network to an SVG file
    test_network2svg(network)

    # Test exporting the network to a DXF file
    test_network2dxf(network)

    # Verify that the generated SVG and DXF files visually match the expected output for a
    # 3x3 grid of circles connected by straight lines (throats).


if __name__ == "__main__":
    main()
