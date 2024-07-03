import sys
import os
import time
import numpy as np
from pathlib import Path

#mod_path = Path("__file__").resolve().parents[2]
#sys.path.append(os.path.abspath(mod_path))

from pore2chip.generate import generate_network
from pore2chip.metrics import get_probability_density
#from pore2chip.src.pore2chip.export import generate_network

def create_random_propterties():
    """
    Generates random numbers used for the pore sizes, throat sizes,
    and coordination numbers using numpy.random.uniform

    Returns:
        properties: Dataset of 3 arrays for each pore property respectively
    """
    pore_diameters = np.random.uniform(low=1.0, high=10.0, size = 20)
    throat_diameters = np.random.uniform(low=0.5, high=6.0, size = 20)
    coordination_nums = np.random.uniform(low=0, high=8, size = 20).astype(np.int64)
    return pore_diameters, throat_diameters, coordination_nums

def test_generate_network(properties, n=5, cc= None):
    """
    Generates an OpenPNM network given pore diameters, throat diameters, and 
    coordination numbers

    Returns:
        network: The generated OpenPNM network
    """
    network = generate_network(n, properties[0], properties[1], properties[2], center_channel=cc)
    print(network)

def test_generate_network2(properties, pdfs, avg_coord=None, n=5, cc= None):
    """
    Generates an OpenPNM network given pore diameters, throat diameters, and 
    coordination numbers and their respective probability densities (from metrics.py)

    Returns:
        network: The generated OpenPNM network
    """
    network = generate_network(n, properties[0], properties[1], properties[2],
                                pdfs[0], pdfs[1], pdfs[2], avg_coord, center_channel=cc)
    print(network)

def main():
    # Generate network
    np.random.seed(seed=1)
    properties = create_random_propterties()

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
    test_generate_network2(properties, [pore_pdf, throat_pdf, coord_pdf], avg_coord=1, n=10, cc=2)

if __name__ == "__main__":
    main()