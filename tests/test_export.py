import sys
import os
import time
import numpy as np
import drawsvg as dr
import math
import ezdxf
import cairosvg
from pathlib import Path

# Set up the module path (I am in tests folder)
mod_path = Path("__file__").resolve().parents[2]
sys.path.append(os.path.abspath(mod_path))

import pore2chip
from pore2chip.src.pore2chip.export import network2svg, network2dxf


def generate_network():
    """
    Generate a simple network with 3x3 pores and straight throats.
    
    Returns:
        dict: A dictionary representing the network structure.
    """
    return {
        "pore.coords": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2),
                        (1, 2), (2, 2)],
        "pore.diameter": [1 for _ in range(9)],
        "throat.conns": [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
                         (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8)],
        "throat.diameter": [0.5 for _ in range(12)]
    }


def test_network2svg(network):
    """
    Test network2svg function with pore_shape='circle' and throat_random=0 (straight throats).
    
    Args:
        network (dict): The network structure.
    """
    svg_result = network2svg(network,
                             n=3,
                             design_size=300,
                             pore_shape='circle',
                             throat_random=0)
    svg_result.save_svg('grain_network.svg')
    cairosvg.svg2png(url="grain_network.svg", write_to="grain_network.png")


def test_network2dxf(network):
    """
    Test network2dxf function with throat_random=0 (straight throats).
    
    Args:
        network (dict): The network structure.
    """
    dxf_result = network2dxf(network, throat_random=0)
    dxf_result.saveas("test_dxf_1.dxf")


def main():
    # Generate network
    network = generate_network()

    # Test SVG export
    test_network2svg(network)

    # Test DXF export
    test_network2dxf(network)

    # Verify that the generated SVG and DXF files visually match the expected output for a
    # 3x3 grid of circles connected by straight lines (throats).


if __name__ == "__main__":
    main()
