import math
import numpy as np
import openpnm as op
import random
from skimage.morphology import diamond
from itertools import chain
 

def generate_network(n1,
                     n2,
                     pore_diameters,
                     throat_diameters,
                     coordination_nums,
                     pore_pdf=None,
                     throat_pdf=None,
                     coord_pdf=None,
                     average_coord=None,
                     min_pore_diameter=None,
                     max_pore_diameter=None,
                     min_throat_diameter=None,
                     max_throat_diameter=None,
                     min_coordination=None,
                     max_coordination=None,
                     pore_random_shift=0.2,
                     lone_pores=True,
                     center_channel=None,
                     return_middle_pores=False,
                     sd=0):
    r"""
    Create 2D OpenPNM network with given pore, throat, and coordination 
    information

    Args:
        n1 (int): Number of desired pores on x-axis
        n2 (int): Number of desired pores on y-axis
        pore_diameters (array): Array of input pore diameters
        pore_pdf (array): Array of pore size probabilities
        throat_diameters (array): Array of input throat diameters
        throat_pdf (array): Array of throat size probabilities
        coordination_nums (array): Array of pore coordination numbers
        coord_pdf (array): Array of pore coordination probabilities
        average_coord (int): Average coordination number (by default, generation 
            will not target specific average coordination number)
        lone_pores (Boolean): Trims pores with no connections (coordination number
            of zero)
        center_channel (int): Number of pores in the center of the network 
            to be connected (allows guaranteed connection from top 
            to bottom)
        return_middle_pores (Boolean): Boolean that indicates if the function 
            returns a tuple with the network and an array of indices 
            of moddle pores for the center channel

    Returns:
        openpnm.models.network : Generated OpenPNM network
    """

    random.seed(sd)
    np.random.seed(sd)

    generated_network = op.network.BodyCenteredCubic([n1, n2, 2])
    op.topotools.trim(generated_network, pores=generated_network.pores('zmax'))
    op.topotools.trim(generated_network,
                      throats=generated_network.throats('body_to_body'))
    op.topotools.trim(generated_network,
                      throats=generated_network.throats('corner_to_corner'))
    generated_network['pore.coords'] *= [1, 1, 0]

    # Add geometry (spheres and cylinders)
    geo = op.models.collections.geometry.spheres_and_cylinders
    generated_network.add_model_collection(geo)
    generated_network.regenerate_models()

    # Shift points and scale (pores reach the edges)
    for pore_index in range(len(generated_network['pore.coords'])):
        generated_network['pore.coords'][pore_index][0] -= 0.5
        generated_network['pore.coords'][pore_index][1] -= 0.5
        generated_network['pore.coords'][pore_index][0] *= (n1 / (n1 - 1))
        generated_network['pore.coords'][pore_index][1] *= (n2 / (n2 - 1))

    # Remove 3D aspects to create 2D image
    del generated_network.params['dimensionality']

    # Get numbers of pores
    num_pores = len(generated_network['pore.coords'])

    # Random sizes and coordination numbers based on extracted data
    # distributions
    random_diameter = None
    temp_coordination = None
    random_coordination = None

    if pore_pdf is not None:
        random_diameter = np.random.choice(a=pore_diameters,
                                           size=num_pores,
                                           replace=True,
                                           p=pore_pdf)
    else:
        random_diameter = random.choices(pore_diameters, k=num_pores)
    if coord_pdf is not None:
        temp_coordination = np.random.choice(coordination_nums,
                                             num_pores,
                                             replace=True,
                                             p=coord_pdf)
        random_coordination = temp_coordination.astype(
            int)  # convert float to int
    else:
        random_coordination = random.choices(coordination_nums, k=num_pores)

    # Assign random pore diameters
    generated_network['pore.diameter'] = random_diameter

    if throat_diameters is None:
        print('Continuing without throats...')
        op.topotools.trim(generated_network,
                          throats=generated_network['throat.all'])
        return generated_network

    # Create list of already visited pores
    visited = []

    # edit connections to get specified coordination numbers
    for pore_index in range(num_pores):

        # Get connected throats
        neighbor_throats = generated_network.find_neighbor_throats(pore_index)

        # If the pore has more throats than what we want...
        if (len(neighbor_throats) > random_coordination[pore_index]
                or neighbor_throats.size > 8):
            if len(neighbor_throats) == 0:
                continue

            neighbor_pores = generated_network.find_neighbor_pores(
                pore_index, flatten=True)

            # Number of connections to remove
            i = len(neighbor_throats) - random_coordination[pore_index]
            # Number of connections that have been removed
            j = 0
            while (i > 0):
                # If we have made the maximum number of throat removals.
                # Otherwise, break the loop
                if j < len(neighbor_pores):
                    # Pick random neighbor throat
                    random_throat = np.random.choice(neighbor_throats)

                    # Makes sure throat index is not above number of throats
                    if random_throat < len(generated_network['throat.conns']):

                        # Connections of the selected throat (tuple of pores the throat connects)
                        conn = generated_network['throat.conns'][random_throat]

                        # If any of the pores in the tuple have not be visited already
                        if conn[0] not in visited or conn[1] not in visited:
                            op.topotools.trim(generated_network,
                                              throats=[random_throat])
                            i -= 1  # One less disconnection that needs to be made

                else:
                    break
                j += 1  # One more disconnection made

        # If the pore has less throats than what we want...
        elif len(neighbor_throats) < random_coordination[pore_index]:
            # Find pores around the vicinity of this pore
            neighbor_pores = generated_network.find_nearby_pores(
                pores=[pore_index], r=1.4, flatten=True)

            # Number of connections to add
            i = random_coordination[pore_index] - len(neighbor_throats)
            # Number of connections made
            j = 0
            while (i > 0
                   ):  # Iterate until all the necessary connections are made
                # If we have made the maximum number of throat connections.
                # Otherwise, break the loop
                if j < len(neighbor_pores):
                    # Pick a random neighbor pore
                    random_pore = np.random.choice(neighbor_pores)

                    # If the pore has not already been visited
                    if random_pore not in visited:
                        # This 'if' statement makes sure that the throat connection is
                        # upper triangular (the first pore index is smaller than the second).
                        # Reduces error messages from OpenPNM
                        if pore_index < random_pore:
                            op.topotools.connect_pores(
                                generated_network, pore_index,
                                random_pore)  # neighbor_pores[-1]
                        else:
                            op.topotools.connect_pores(generated_network,
                                                       random_pore, pore_index)
                        i -= 1  # One less connection that needs to be made
                else:
                    break
                j += 1  # One more connection made

        # Finished assigning coordination to pore. Now we add it to visited pores
        visited.append(pore_index)

    ##### Middle Throats #####
    # Getting middle pores
    middle_pores = []
    if center_channel is not None:
        k = center_channel  # middle channel width (4)
        j = math.floor((n1 - center_channel) / 2)  # outer channel width (3)

        initial_pore = j * n2

        for index in range(initial_pore, initial_pore + (k * n2)):
            middle_pores.append(index)

        for index in range(initial_pore + (n1 * n2) - j,
                           initial_pore + (n1 * n2) - j + (k * (n2 - 1))):
            middle_pores.append(index)

        current_pore = middle_pores[0]
        next_pore = None

        for i in range(n2 + (n2 - 1)):
            neighbor_pores = generated_network.find_nearby_pores(
                pores=[current_pore], r=1, flatten=True)
            next_pore_list = []
            for ind in neighbor_pores:
                if ind in middle_pores and generated_network['pore.coords'][
                        ind][1] > generated_network['pore.coords'][
                            current_pore][1]:
                    next_pore_list.append(ind)
            if len(next_pore_list) != 0:
                next_pore = random.choice(next_pore_list)

                connected_pores = generated_network.find_neighbor_pores(
                    [current_pore], flatten=True)
                if next_pore not in connected_pores:
                    if current_pore < next_pore:
                        op.topotools.connect_pores(generated_network,
                                                   current_pore,
                                                   next_pore,
                                                   labels=['middle'])
                    else:
                        op.topotools.connect_pores(generated_network,
                                                   next_pore,
                                                   current_pore,
                                                   labels=['middle'])
                #print('current:',current_pore, 'next:', next_pore)
                current_pore = next_pore

    # Remove any duplicate throats that may have been formed
    dupes = op.models.network.duplicate_throats(generated_network)
    op.topotools.trim(generated_network, throats=dupes)

    # Zero Coordination Fixes
    coord = op.models.network.coordination_number(generated_network)
    # If there is no coordination value of 0 in our random selection
    # but we still see some pores with a coordination of 0,
    # connect them with a neighbor
    if 0 not in random_coordination and 0 in coord:
        indicies = list(chain.from_iterable(np.where(coord == 0)))
        for ind in indicies:
            neighbor_pores = generated_network.find_nearby_pores(
                pores=ind, r=1.5, flatten=True)
            pore_to_connect = None
            for neighbor in neighbor_pores:
                coordination = len(generated_network.find_neighbor_throats(neighbor))
                if coordination >= max(random_coordination):
                    continue
                else:
                    pore_to_connect = neighbor
            if pore_to_connect is None:
                pore_to_connect = neighbor_pores[0]
            if ind < pore_to_connect:
                op.topotools.connect_pores(generated_network, ind, pore_to_connect)
            else:
                op.topotools.connect_pores(generated_network, pore_to_connect, ind)

    # Higher Coordination Fixes
    if max(coord) > max(random_coordination):
        max_possible = max(random_coordination)
        indicies = list(chain.from_iterable(np.where(coord > max_possible)))
        for ind in indicies:
            neighbor_pores = generated_network.find_neighbor_pores(ind)
            current_coord = len(neighbor_pores)
            i = current_coord - max_possible
            j = 0
            while(i > 0):
                if j > (current_coord - max_possible):
                    print('Failed to change higher coordination. Continuing...')
                    break
                neighbor = neighbor_pores[j]
                print('neighbor:', neighbor)
                neighbors_throats = generated_network.find_neighbor_throats(neighbor)
                if len(neighbors_throats) > min(random_coordination):
                    if ind < neighbor:
                        op.topotools.trim(generated_network,
                                            throats=[[ind, neighbor]])
                    else:
                        op.topotools.trim(generated_network,
                                            throats=[[neighbor, ind]])
                    i -= 1
                j += 1

    # Remove non-connected pores if flag is true
    if not lone_pores:
        op.topotools.trim(generated_network, pores=np.where(coord == 0))

    # Reduce even further to an average coordination
    if average_coord is not None:
        reduce = op.topotools.reduce_coordination(generated_network,
                                                  average_coord)
        op.topotools.trim(generated_network, throats=reduce)

    # Slightly randomize pore positions
    for pore_index in range(len(generated_network['pore.coords'])):
        shift_amount_x = np.random.uniform(-pore_random_shift,
                                           pore_random_shift)
        shift_amount_y = np.random.uniform(-pore_random_shift,
                                           pore_random_shift)
        generated_network['pore.coords'][pore_index][0] += shift_amount_x
        generated_network['pore.coords'][pore_index][1] += shift_amount_y

    # Assign random pore throat diameters
    num_throats = len(generated_network['throat.conns'])
    random_throat_diameter = None
    if throat_pdf is not None:
        random_throat_diameter = np.random.choice(throat_diameters,
                                                  num_throats,
                                                  replace=True,
                                                  p=throat_pdf)
    else:
        random_throat_diameter = random.choices(throat_diameters,
                                                k=num_throats)

    generated_network['throat.diameter'] = random_throat_diameter

    # Assign minimum and maximum throat diameters
    if min_pore_diameter is not None:
        generated_network['pore.diameter'][np.where(
            generated_network['pore.diameter'] <
            min_pore_diameter)] = min_pore_diameter
    if max_pore_diameter is not None:
        generated_network['pore.diameter'][np.where(
            generated_network['pore.diameter'] >
            max_pore_diameter)] = max_pore_diameter
    if min_throat_diameter is not None:
        generated_network['throat.diameter'][np.where(
            generated_network['throat.diameter'] <
            min_throat_diameter)] = min_throat_diameter
    if max_throat_diameter is not None:
        generated_network['throat.diameter'][np.where(
            generated_network['throat.diameter'] >
            max_throat_diameter)] = max_throat_diameter

    if return_middle_pores:
        return generated_network, middle_pores
    else:
        return generated_network
