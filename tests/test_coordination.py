import sys
import os
import time
import openpnm as op
import porespy as ps
from pathlib import Path

# Uncomment and modify these lines if the pore2chip package is not in the PYTHONPATH
#mod_path = Path("__file__").resolve().parents[2]
#sys.path.append(os.path.abspath(mod_path))

from pore2chip.coordination import coordination_nums_2D, coordination_nums_3D
#from pore2chip.src.pore2chip.export import coordination_nums_3D, coordination_nums_2D

def generate_network():
    """
    Generate a simple network with 3x3 pores and straight throats.
    
    Returns:
        network (dict): A dictionary representing the network structure. An OpenPNM Cubic network object configured with spheres and cylinders geometry models.
    """
    network = op.network.Cubic([3, 3]) # Create a 3x3 cubic network
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders) # Add geometry models
    network.regenerate_models() # Generate network properties
    print(network) # For debugging purposes
    return network

def test_coordination_nums_3D(network):
    """
    Test coordination_nums_2D function given an OpenPNM network.
    
    Args:
        network (dict): The network structure and coordination numbers (e.g., op.network.Cubic).

    Returns:
        coordination_numbers (ndarray): An array of coordination numbers for each pore in the network..
    """
    coordination_numbers = coordination_nums_3D(pn=network)
    print(coordination_numbers) # For debugging purposes
    print(coordination_numbers.mean()) # For debugging purposes
    print(coordination_numbers.shape) # For debugging purposes
    return coordination_numbers

def test_coordination_nums_2D(img_list):
    """
    Test coordination_nums_2D function given a list of images.
    
    Args:
        img_list (ndarray): 3D image list

    Returns:
        coordination_numbers (ndarray): An array of coordination numbers for each pore in the network..
    """
    coordination_numbers = coordination_nums_2D(img_list)
    print(coordination_numbers)
    print(coordination_numbers.mean())
    print(coordination_numbers.shape)
    return coordination_numbers

def main():
    """
    Main function to generate a network, test coordination number computations on 3D networks
    and 2D image slices.
    """

    # Generate network
    network = generate_network()

    # Test coordination 3D on generated network
    test_coordination_nums_3D(network)

    # Generate a 3D image using PoreSpy with specified dimensions and random seed
    blobs = ps.generators.blobs([5, 100, 100], seed=1)

    # Test coordination 2D on image list
    test_coordination_nums_2D(blobs)


if __name__ == "__main__":
    main()