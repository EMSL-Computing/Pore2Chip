import sys
import os
import time
import numpy as np
from pathlib import Path

# Uncomment and modify these lines if the pore2chip package is not in the PYTHONPATH
#mod_path = Path("__file__").resolve().parents[2]
#sys.path.append(os.path.abspath(mod_path))

# Importing functions from the pore2chip package
from pore2chip.generate import generate_network
from pore2chip.metrics import get_probability_density


def create_random_properties():
    """
    Generates random numbers used for the pore sizes, throat sizes,
    and coordination numbers using numpy.random.uniform

    Returns:
        tuple: Dataset of 3 arrays for each pore property respectively (pore diameters, throat diameters, and coordination numbers).
            - pore_diameters (np.ndarray): Array of random pore diameters.
            - throat_diameters (np.ndarray): Array of random throat diameters.
            - coordination_nums (np.ndarray): Array of random coordination numbers (converted to integers).
    """
    # Generating random pore diameters within specified range
    pore_diameters = np.random.uniform(low=1.0, high=10.0, size=20, seed=1)

    # Generating random throat diameters within specified range
    throat_diameters = np.random.uniform(low=0.5, high=6.0, size=20, seed=1)

    # Generating random coordination numbers within specified range and converting to integers
    coordination_nums = np.random.uniform(low=0, high=8,
                                          size=20, seed=1).astype(np.int64)
    return pore_diameters, throat_diameters, coordination_nums


def test_generate_network(properties, n=5, cc=None):
    """
    Generates an OpenPNM network given pore diameters, throat diameters, and 
    coordination numbers

    Args:
    properties (tuple): A tuple containing pore diameters, throat diameters, and coordination numbers.
        - pore diameters
        - throat diameters
        - coordination numbers
    n (int, optional): The dimension for the network generation, default is 5.
    cc (int, optional): Center channel size, if any, defaults to None

    Returns:
        network: The generated OpenPNM network
    """
    # Generating the network with provided dimensions and properties
    network = generate_network(n,
                               n,
                               properties[0],
                               properties[1],
                               properties[2],
                               center_channel=cc)

    # Outputting the network details for visual inspection
    print(network)


def test_generate_network2(properties, pdfs, avg_coord=None, n=5, cc=None):
    """
    Generates an OpenPNM network given pore diameters, throat diameters, and 
    coordination numbers and their respective probability densities (from metrics.py)

    Args:
        properties (tuple): A tuple containing pore diameters, throat diameters, and coordination numbers.
        pdfs (list): A list containing probability density functions for each property.
        avg_coord (int, optional): Average coordination number to be used, if provided.
        n (int): The dimension for the network generation, default is 5.
        cc (int, optional): Center channel size, if any.

    Returns:
        network: The generated OpenPNM network
    """

    # Generating the network with additional parameters including probability densities
    network = generate_network(n,
                               n,
                               properties[0],
                               properties[1],
                               properties[2],
                               pdfs[0],
                               pdfs[1],
                               pdfs[2],
                               avg_coord,
                               center_channel=cc)

    # Outputting the network details
    print(network)


def main():
    """
    Main function to generate properties for a network, create and test network generation with different parameters.
    """

    # Generate network; Setting a random seed for reproducibility
    np.random.seed(seed=1)

    # Generating random properties for network elements
    properties = create_random_properties()

    # Test generation (n=5, no center channel)
    test_generate_network(properties)

    # Test generation (n=10, center channel of 3 nodes)
    test_generate_network(properties, 10, 3)

    ### Advanced Tests ###

    # Getting probability densities for each array of values (all add up to 1.0)
    # These probability weights will be used instead of randomly sampling the array
    pore_pdf = get_probability_density(properties[0])
    throat_pdf = get_probability_density(properties[1])
    coord_pdf = get_probability_density(properties[2])

    # Test generation (n=5, center channel of 2 nodes)
    test_generate_network2(properties, [pore_pdf, throat_pdf, coord_pdf], cc=2)

    # Test generation (n=10, center channel of 2 nodes, average coordination of 1)
    test_generate_network2(properties, [pore_pdf, throat_pdf, coord_pdf],
                           avg_coord=1,
                           n=10,
                           cc=2)


if __name__ == "__main__":
    main()
