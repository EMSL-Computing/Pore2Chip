import numpy as np
import cv2 as cv
import copy

def _write_vtk(vtk_fl_name, num_nodes, num_elements, nodes_per_element, \
                    coord_matrix, connectivity_matrix, mat_id_matrix, **kwargs):
    r"""
    Helper function for img2vtk()
    """

    fid = open(vtk_fl_name + '.vtk', 'w+')

    # Write the 'Header' for ASCII files; HEADER: Lines required in
    # output file = '5'
    fid.write('# vtk DataFile Version 2.0 \n')
    fid.write('Written using Python (TEST CASE) \n')
    fid.write('ASCII \n')
    fid.write('DATASET UNSTRUCTURED_GRID \n \n')

    # Write the coordinate matrix -- Relative
    fid.write('POINTS ' + str(int(num_nodes)) + ' float \n')
    for i in range(0, num_nodes):
        fid.write(str(float(coord_matrix[i,0])) + ' ' + \
                    str(float(coord_matrix[i,1])) + ' ' + \
                    str(float(coord_matrix[i,2])) + '\n')
    fid.write('\n')

    # Write the connectivity matrix
    fid.write('CELLS ' + str(int(num_elements)) + ' ' + \
              str(int(num_elements*(nodes_per_element+1))) + ' \n')
    for i in range(0, num_elements):
        fid.write(str(8) + ' ' + str(int(connectivity_matrix[i,0])) + \
                  ' ' + str(int(connectivity_matrix[i,1])) + \
                  ' ' + str(int(connectivity_matrix[i,2])) + \
                  ' ' + str(int(connectivity_matrix[i,3])) + \
                  ' ' + str(int(connectivity_matrix[i,4])) + \
                  ' ' + str(int(connectivity_matrix[i,5])) + \
                  ' ' + str(int(connectivity_matrix[i,6])) + \
                  ' ' + str(int(connectivity_matrix[i,7])) + '\n')
    fid.write('\n')

    # Write cell types VTK_HEXAHEDRON (=12)
    fid.write('CELL_TYPES ' + str(int(num_elements)) + ' \n')
    for i in range(0, num_elements):
        fid.write(str(12) + '\n')
    fid.write('\n')

    fid.write('CELL_DATA ' + str(int(num_elements)) + ' \n')

    # Write extra kwarg arguments
    for key, val in kwargs.items():
        print("Writing %s..." % key)
        fid.write('SCALARS ' + key + ' FLOAT \n')
        fid.write('LOOKUP_TABLE default \n')
        for i in range(0, num_elements):
            fid.write(str(float(val[i])) + '\n')
        fid.write('\n')

    # Write material ID
    fid.write('SCALARS Material-ID FLOAT \n')
    fid.write('LOOKUP_TABLE default \n')
    for i in range(0, num_elements):
        fid.write(str(float(mat_id_matrix[i])) + '\n')

    fid.write('\n')
    fid.close()
    return


def img2vtk(image, vtk_fl_name, dims, **kwargs):
    r"""
        Converts 2D image array to 3D vtk model

      Args:
        image (2D array): 2D image array
        vtk_fl_name (str): Filename/filepath
        dims (array): Size 3 Array-like for x, y, and z dimentions respectively
        **kwargs : Extra arguements to write to VTK file.
            Examples: Permeability = perm_matrix ... 
            where perm_matrix is an array defining permeability
            values for each pixel

        generated_network (dict): The OpenPNM network containing pore and throat information.
        throat_random (int): Multiplier that decides how random the throat shape will be. Default is 1.
            A value of 0 makes the throats straight.

      Returns:
        None
    """
    xseed = image.shape[0]
    yseed = image.shape[1]
    depth = dims[2]
    img_3d_stack = np.zeros((yseed, dims[2], xseed), dtype=int)
    for i in range(dims[2]):
        img_3d_stack[:, i, :] = copy.deepcopy(image)

    num_elements = (depth - 1) * (xseed - 1) * (yseed - 1)
    num_nodes = (depth) * (xseed) * (yseed)

    Lx = dims[0] / 1000
    Ly = dims[1] / 1000
    Lz = dims[2] / 1000

    xcoord_list = np.linspace(0, 0.1 * Lx, xseed)
    ycoord_list = np.linspace(0, 0.1 * Ly, yseed)
    zcoord_list = np.linspace(0, 0.1 * Lz, depth)

    #Right-handed coordinate system
    coord_list = np.ndarray(shape=(num_nodes, 3), dtype=float)

    for k in range(0, depth):  #Z-dir
        for j in range(0, yseed):  #Y-dir
            for i in range(0, xseed):  #X-dir
                xcoord = xcoord_list[i]
                ycoord = ycoord_list[j]
                zcoord = zcoord_list[k]

                node = i + j * xseed + k * xseed * yseed
                coord_list[node, 0] = xcoord
                coord_list[node, 1] = ycoord
                coord_list[node, 2] = zcoord

    #Right-handed coordinate system
    #Create connectivity matrix -- Structured grid
    connectivity_list = np.ndarray(shape=(num_elements, 8), dtype=int)
    counter = 0
    for k in range(0, depth - 1):  # Create connectivity matrix
        for j in range(0, yseed - 1):
            for i in range(0, xseed - 1):
                # Node number
                index = i + j * xseed + k * xseed * yseed
                # Cell number
                elem_index = i + j * (xseed - 1) + k * (xseed - 1) * (yseed -
                                                                      1)

                connectivity_list[counter, 0] = index
                connectivity_list[counter, 1] = index + 1
                connectivity_list[counter, 2] = index + 1 + xseed
                connectivity_list[counter, 3] = index + xseed
                connectivity_list[counter, 4] = index + xseed * yseed
                connectivity_list[counter, 5] = index + 1 + xseed * yseed
                connectivity_list[counter,
                                  6] = index + 1 + xseed + xseed * yseed
                connectivity_list[counter, 7] = index + xseed + xseed * yseed

                counter = counter + 1

    # Material ids for h5 and VTK
    th3_id_list = np.ndarray(shape=(num_elements, 1), dtype=int)

    counter = 0
    for k in range(0, depth - 1):
        for j in range(0, yseed - 1):  # Y-dir
            for i in range(0, xseed - 1):  # X-dir
                # Cell number
                elem_index = i + j * (xseed - 1) + k * (xseed - 1) * (yseed -
                                                                      1)
                th3_id_list[elem_index, 0] = img_3d_stack[j, k, i]

    nodes_per_element = 8

    _write_vtk(vtk_fl_name, num_nodes, num_elements, nodes_per_element, \
                    coord_list, connectivity_list, th3_id_list, **kwargs)

    print("Finished writing to: %s.vtk" % vtk_fl_name)
    return
