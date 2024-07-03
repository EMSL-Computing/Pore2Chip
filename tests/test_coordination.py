import sys
import os
import time
import openpnm as op
import porespy as ps
from pathlib import Path

#mod_path = Path("__file__").resolve().parents[2]
#sys.path.append(os.path.abspath(mod_path))

from pore2chip.coordination import coordination_nums_2D, coordination_nums_3D
#from pore2chip.src.pore2chip.export import coordination_nums_3D, coordination_nums_2D

def generate_network():
    """
    Generate a simple network with 3x3 pores and straight throats.
    
    Returns:
        dict: A dictionary representing the network structure.
    """
    network = op.network.Cubic([3, 3])
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    network.regenerate_models()
    print(network)
    return network

def test_coordination_nums_3D(network):
    """
    Test coordination_nums_2D function given an OpenPNM network.
    
    Args:
        network (dict): The network structure.
    """
    coordination_numbers = coordination_nums_3D(pn=network)
    print(coordination_numbers)
    print(coordination_numbers.mean())
    return coordination_numbers

def test_coordination_nums_2D(img_list):
    """
    Test coordination_nums_2D function given a list of images.
    
    Args:
        img_list (ndarray): 3D image list
    """
    coordination_numbers = coordination_nums_2D(img_list)
    print(coordination_numbers)
    print(coordination_numbers.mean())
    return coordination_numbers

def main():
    # Generate network
    network = generate_network()

    # Test coordination 3D on network
    test_coordination_nums_3D(network)

    # Generate a 3D image using PoreSpy
    blobs = ps.generators.blobs([5, 100, 100], seed=1)

    # Test coordination 2D on image list
    test_coordination_nums_2D(blobs)


if __name__ == "__main__":
    main()