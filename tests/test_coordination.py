import sys
import os
import time
import openpnm as op
import porespy as ps
from pathlib import Path
import pytest
import numpy as np

# Uncomment and modify these lines if the pore2chip package is not in the PYTHONPATH
#mod_path = Path("__file__").resolve().parents[2]
#sys.path.append(os.path.abspath(mod_path))

from pore2chip.coordination import coordination_nums_2D, coordination_nums_3D

def generate_network():
    """
    Generate a simple network with 3x3 pores and straight throats.
    
    Returns:
        network (dict): A dictionary representing the network structure. An OpenPNM Cubic network object configured with spheres and cylinders geometry models.
    """
    network = op.network.Cubic([3, 3])  # Create a 3x3 cubic network
    network.add_model_collection(op.models.collections.geometry.
                                spheres_and_cylinders)  # Add geometry models
    network.regenerate_models()  # Generate network properties
    print(network)  # For debugging purposes
    return network

class TestCoordination():

    def test_coordination_nums_3D(self):
        """
        Test coordination_nums_2D function given an OpenPNM network.
    
        Args:
            network (dict): The network structure and coordination numbers (e.g., op.network.Cubic).

        Returns:
            coordination_numbers (ndarray): An array of coordination numbers for each pore in the network..
        """
        network = generate_network()
        coordination_numbers = coordination_nums_3D(pn=network)
        assert coordination_numbers.any() == np.array([2,3,2,3,4,3,2,3,2],dtype=int).any()
        #print(coordination_numbers)  # For debugging purposes
        #print(coordination_numbers.mean())  # For debugging purposes
        #print(coordination_numbers.shape)  # For debugging purposes
        #return coordination_numbers


    def test_coordination_nums_2D(self):
        """
        Test coordination_nums_2D function given a list of images.
    
        Args:
            img_list (ndarray): 3D image list

        Returns:
            coordination_numbers (ndarray): An array of coordination numbers for each pore in the network..
        """
        blobs = ps.generators.blobs([1, 100, 100], seed=1)
        coordination_numbers = coordination_nums_2D(blobs)
        assert coordination_numbers.any() == np.array(
            [3, 3, 1, 0, 3, 1, 0, 1, 1, 3, 2, 2, 3, 3, 1, 2, 3, 2, 2, 1, 2, 1, 3, 1,      
             2, 2, 5, 2, 0, 0, 1, 3, 1, 2, 1, 0, 1, 3, 4, 0, 3, 3, 0, 1, 2, 2, 3, 3,
             2, 2, 2, 2, 3, 3, 0, 4, 1, 2, 4, 2, 1, 2, 3, 3, 2, 3, 5, 4, 1, 1, 2, 1,
             2, 1, 2, 3, 3, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,]
            ,dtype=float).any()
        #print(coordination_numbers)
        #print(coordination_numbers.mean())
        #print(coordination_numbers.shape)
        #return coordination_numbers


#def main():
    #"""
    #Main function to generate a network, test coordination number computations on 3D networks
    #and 2D image slices.
    #"""

    ## Generate network
    #network = generate_network()

    ## Test coordination 3D on generated network
    #test_coordination_nums_3D(network)

    ## Generate a 3D image using PoreSpy with specified dimensions and random seed
    #blobs = ps.generators.blobs([5, 100, 100], seed=1)

    ## Test coordination 2D on image list
    #test_coordination_nums_2D(blobs)


#if __name__ == "__main__":
    #main()
