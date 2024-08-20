"""
Functions to create SVG and DXF files from network .
"""

import numpy as np
import drawsvg as dr
import math
import ezdxf


def network2svg(
    generated_network,  # An OpenPNM network object
    n1,  # Number of pores on x-axis
    n2,  # Number of pores on y-axis
    d1,  # x dimension of the SVG image
    d2,  # y dimension of the SVG image
    pore_shape='blob',  # Shape of the pores ('blob' or 'circle')
    throat_random=1,  # Multiplier for throat randomness (0: straight, 1: random)
    pore_debug=False,  # Boolean to draw blue circles for pores (debugging)
    throat_debug=False,  # Boolean to draw red lines for throats (debugging)
    throat_random_debug=False,
    no_throats=False,
    middle_pores=None,
    disconnected=None
):  # Boolean to draw perp. lines for throat placement (debugging)
    r"""
    Create an SVG file from an OpenPNM network. 
    The network2svg function converts an OpenPNM network into an SVG image. 
    It represents pores as either blobs or circles, and throats as either simple lines 
    or a series of connected circles to form a more complex shape

    Parameters:
        generated_network : dict
            The OpenPNM network containing pore and throat information.
        n1 : int
            Number of pores on the x-axis of the model grid
        n2 : int
            Number of pores on the y-axis of the model grid
        d1 : int
            The x length of the SVG image (in pixels).
        d2 : int
            The y length of the SVG image (in pixels).
        pore_shape : str, optional
            Shape of the pore bodies, can be 'blob' or 'circle'. Default is 'blob'.
        throat_random : int, optional
            Multiplier that decides how random the throat shape will be. Default is 1.
            A value of 0 makes the throats straight.
        pore_debug : bool, optional
            If True, draws pore bodies as blue and draws pore index number next 
            to the pore. Default is False.
        throat_debug : bool, optional
            If True, draws lines where throats connect. Default is False.
        throat_random_debug : bool, optional
            If True, draws lines perpendicular to throat directions used to randomize
            throat shape. `throat_debug` must be True. Default is False.
        middle_pores : array-like
            Array or pore indices that indicate the center channel for debugging
            Renders the middle pores in green. 
            Also writes pore index number next to dot. Default is None.

    Returns:
        dr.Drawing
            A drawSvg drawing object representing the network.
    """

    # Initialize the drawing with the specified size and origin at the bottom left
    design = dr.Drawing(d1, d2, origin=(0, 0))

    # Get the number of pores from the network
    num_pores = len(generated_network['pore.coords'])

    # Draw each pore based on the specified shape
    if pore_shape == 'blob':
        # Draw each pore as random blobs
        fill_color = 'black'  # Default fill color for pores
        if pore_debug:
            fill_color = 'blue'  # Change fill color to blue for debugging
        for pore_index in range(num_pores):
            if disconnected is not None:
                if pore_index in disconnected:
                    fill_color = 'red'
            x_coord = generated_network['pore.coords'][pore_index][0] * (d1 /
                                                                         n1)
            y_coord = generated_network['pore.coords'][pore_index][1] * (d2 /
                                                                         n2)

            # Create a path object to define the pore shape and
            # adjust y-coordinate to match OpenPNM network (drawsvg has different origin)
            if middle_pores is not None and pore_index in middle_pores:
                p = dr.Path(fill='green',
                            fill_opacity=1.0,
                            close=True,
                            transform='translate(' + str(x_coord) + ',' +
                            str((-y_coord) + d2) + ')')
            else:
                p = dr.Path(fill=fill_color,
                            fill_opacity=1.0,
                            close=True,
                            transform='translate(' + str(x_coord) + ',' +
                            str((-y_coord) + d2) + ')')

            positions = []
            radius = generated_network['pore.diameter'][pore_index] / 2

            # Generate random positions for Bézier curve control points
            for i in range(9):
                rand_radius = radius + np.random.uniform(
                    -0.4 * radius, 0.4 * radius)
                positions.append({
                    'x':
                    math.cos((math.pi / 4) * i) * radius,
                    'y':
                    math.sin((math.pi / 4) * i) * radius,
                    'mx':
                    math.cos((math.pi / 4) * i - math.radians(20)) *
                    rand_radius,
                    'my':
                    math.sin((math.pi / 4) * i - math.radians(20)) *
                    rand_radius
                })

                # Define Bézier curves to create the blob shape using Move (M), Quadratic Bézier (Q),
                # and Cubic Bézier (T) commands from the drawsvg library
                if i == 0:
                    p.M(positions[i]['x'],
                        positions[i]['y'])  # Move to starting point
                elif i == 1:
                    p.Q(positions[i]['mx'], positions[i]['my'],
                        positions[i]['x'],
                        positions[i]['y'])  # Quadratic Bézier curve
                else:
                    p.T(positions[i]['x'],
                        positions[i]['y'])  # Cubic Bézier curve

            design.append(p)  # Add the pore path to the SVG design
            if pore_debug:
                design.append(
                    dr.Text(str(pore_index),
                            6,
                            x_coord,
                            -y_coord + d2,
                            fill=fill_color))

    # This section handles the case where pore_shape is 'circle'
    elif pore_shape == 'circle':
        # Draw each pore as circles
        fill_color = 'black'  # Default fill color for pores
        if pore_debug:
            fill_color = 'blue'  # Change fill color to blue for debugging
        for pore_index in range(num_pores):
            if disconnected is not None:
                if pore_index in disconnected:
                    fill_color = 'red'
            x_coord = generated_network['pore.coords'][pore_index][0] * (d1 /
                                                                         n1)
            y_coord = generated_network['pore.coords'][pore_index][1] * (d2 /
                                                                         n2)
            radius = generated_network['pore.diameter'][pore_index] / 2
            # Create a circle object using the drawsvg library
            # Adjust y-coordinate to match OpenPNM network (drawsvg has different origin)
            if middle_pores is not None and pore_index in middle_pores:
                design.append(
                    dr.Circle(x_coord, (-y_coord) + d2,
                              radius / 2,
                              fill='green'))
                if pore_debug:
                    design.append(
                        dr.Text(str(pore_index),
                                6,
                                x_coord,
                                -y_coord + d2,
                                fill='green'))
            else:
                design.append(
                    dr.Circle(x_coord, (-y_coord) + d2,
                              radius / 2,
                              fill=fill_color))
                if pore_debug:
                    design.append(
                        dr.Text(str(pore_index),
                                6,
                                x_coord,
                                -y_coord + d2,
                                fill='blue'))
    else:
        print(
            'Error: Invalid shape for pore body (Must be \'blob\' or \'circle\')'
        )
        return None  # This section handles an invalid pore_shape argument (anything other than 'blob' or 'circle')

    # Check if 'throat.conns' data exists in the network
    if generated_network.get(
            'throat.conns') is not None and no_throats == False:
        # Get the number of throats
        #num_throats = len(generated_network['throat.all']) #This might be a bug
        num_throats = len(generated_network['throat.conns'])

        for throat_index in range(num_throats):
            # Get the indices of the pores connected by the throat
            pore1 = generated_network['throat.conns'][throat_index][0]
            pore2 = generated_network['throat.conns'][throat_index][1]

            # Get the coordinates of the connected pores
            pore1_x = generated_network['pore.coords'][pore1][0] * (d1 / n1)
            pore1_y = generated_network['pore.coords'][pore1][1] * (d2 / n2)
            pore2_x = generated_network['pore.coords'][pore2][0] * (d1 / n1)
            pore2_y = generated_network['pore.coords'][pore2][1] * (d2 / n2)

            # Convert coordinates to match the SVG coordinate system
            pore1_coords = [pore1_x, (-pore1_y) + d2]
            pore2_coords = [pore2_x, (-pore2_y) + d2]

            # Distance between the two pores
            distance = math.dist(pore1_coords, pore2_coords)

            # Skip if the throat diameter is not defined
            if math.isnan(generated_network['throat.diameter'][throat_index]):
                continue

            # Calculate the number of points (circles) in the throat
            num_throat_points = math.ceil(
                distance /
                (generated_network['throat.diameter'][throat_index]))
            num_throat_points += round(num_throat_points / 2)

            # Calculate the segment distances and magnitude
            x_segment = (pore2_coords[0] - pore1_coords[0]) / num_throat_points
            y_segment = (pore2_coords[1] - pore1_coords[1]) / num_throat_points
            magnitude = 2 * (
                generated_network['throat.diameter'][throat_index] / 2)

            # Loop to create each circle representing the throat
            for i in range(num_throat_points):
                # Base point for each circle
                base_point = [
                    pore1_coords[0] + (x_segment * i),
                    pore1_coords[1] + (y_segment * i)
                ]

                # Randomly decide the direction of the perpendicular vector
                direction = np.random.randint(0, 2)
                perp_vector = [0, 0]

                if direction == 0:
                    # Set perpendicular vector in the y-direction
                    perp_vector = [
                        y_segment / magnitude, -x_segment / magnitude
                    ]
                else:
                    # Set perpendicular vector in the x-direction
                    perp_vector = [
                        -y_segment / magnitude, x_segment / magnitude
                    ]

                # Random shift based on throat diameter and throat_random parameter
                throat_radius = generated_network['throat.diameter'][
                    throat_index] / 2
                random_shift = np.random.uniform(-throat_radius,
                                                 throat_radius) * throat_random
                perp_vector[0] *= random_shift
                perp_vector[1] *= random_shift

                # Calculate new position for the circle
                x_new = base_point[0] + perp_vector[0]
                y_new = base_point[1] + perp_vector[1]

                # Add the circle to the design
                throat_fill = 'black'
                if disconnected is not None:
                    if pore1 in disconnected or pore2 in disconnected:
                        throat_fill = 'red'

                design.append(
                    dr.Circle(x_new,
                              y_new,
                              throat_radius,
                              fill=throat_fill,
                              fill_opacity=1.0))

        # Draw lines for debugging throats (visible only if throat_debug is True)
        if throat_debug:
            for throat_index in range(num_throats):
                # Get pore indices connected by the throat
                pore1 = generated_network['throat.conns'][throat_index][0]
                pore2 = generated_network['throat.conns'][throat_index][1]

                # Get coordinates of the connected pores
                pore1_x = generated_network['pore.coords'][pore1][0] * (d1 /
                                                                        n1)
                pore1_y = generated_network['pore.coords'][pore1][1] * (d2 /
                                                                        n2)
                pore2_x = generated_network['pore.coords'][pore2][0] * (d1 /
                                                                        n1)
                pore2_y = generated_network['pore.coords'][pore2][1] * (d2 /
                                                                        n2)

                # Adjust coordinates for the SVG coordinate system
                pore1_coords = [pore1_x, (-pore1_y) + d2]
                pore2_coords = [pore2_x, (-pore2_y) + d2]

                # Draw a red line to represent the connection between pores (throat)
                design.append(
                    dr.Line(pore1_coords[0],
                            pore1_coords[1],
                            pore2_coords[0],
                            pore2_coords[1],
                            stroke='red',
                            stroke_opacity=0.5,
                            stroke_width=1))

                # Additional debugging for throat_random (visible only if throat_random_debug is True)
                if throat_random_debug:
                    for i in range(num_throat_points):
                        # Base point for the current circle position
                        base_point = [
                            pore1_coords[0] + (x_segment * i),
                            pore1_coords[1] + (y_segment * i)
                        ]

                        # Calculate the position with the random perpendicular vector applied
                        x_new = base_point[0] + perp_vector[0]
                        y_new = base_point[1] + perp_vector[1]

                        # Draw a red line to show the position after applying the random shift
                        design.append(
                            dr.Line(base_point[0],
                                    base_point[1],
                                    x_new,
                                    y_new,
                                    stroke='red',
                                    stroke_width=1))

    return design


def network2dxf(
    generated_network,
    throat_random=1,
    no_throats=False,
):
    r"""
    Create a DXF file from an OpenPNM network.
    The network2dxf function generates a DXF file from an OpenPNM network. 
    It represents pores as circles and throats as a series of connected circles 
    to form irregular shapes, with an optional parameter to introduce randomness 
    in the throat shapes.

    Parameters:
        generated_network : dict
            The OpenPNM network containing pore and throat information.
        throat_random : int, optional
            Multiplier that decides how random the throat shape will be. Default is 1.
            A value of 0 makes the throats straight.

    Returns:
        ezdxf.document
            An ezdxf document object representing the network.
    """

    # Create a new DXF document with DXF version R2000
    document = ezdxf.new(dxfversion="R2000")

    # Get the modelspace (the main drawing area) from the document
    modelspace = document.modelspace()

    # Create a hatch object (used to fill closed shapes) with color 7 (default gray)
    hatch = modelspace.add_hatch(color=7)

    # Get the number of pores in the network
    num_pores = len(generated_network['pore.coords'])

    # Iterate over each pore to add it to the model space
    for pore_index in range(num_pores):
        # Extract pore coordinates (x and y) from the network data
        x_coord = generated_network['pore.coords'][pore_index][0]
        y_coord = generated_network['pore.coords'][pore_index][1]

        # Calculate pore radius by dividing the diameter by a scaling factor (20 in this case)
        radius = generated_network['pore.diameter'][pore_index] / 20

        # Add a circle to the modelspace representing the current pore
        modelspace.add_circle((x_coord, y_coord), radius=radius)

        # Create a new hatch object for the pore (to define its fill)
        hatch = modelspace.add_hatch(color=7)

        # Add an edge path to the hatch object (defines the boundary of the filled area)
        edge_path = hatch.paths.add_edge_path()

        # Add an ellipse to the edge path (representing the circular pore with same radius)
        edge_path.add_ellipse((x_coord, y_coord),
                              major_axis=(0, radius),
                              ratio=1)

    # Check if throat connections exist in the network
    if generated_network.get('throat.conns') is not None:
        if no_throats == False:
            # Get the number of throats in the network
            num_throats = len(generated_network['throat.conns'])

            # Loop through each throat in the network
            for throat_index in range(num_throats):
                # Get the indices of the pores connected by the throat
                pore1 = generated_network['throat.conns'][throat_index][0]
                pore2 = generated_network['throat.conns'][throat_index][1]

                # Get the coordinates of the connected pores
                pore1_x = generated_network['pore.coords'][pore1][0]
                pore1_y = generated_network['pore.coords'][pore1][1]
                pore2_x = generated_network['pore.coords'][pore2][0]
                pore2_y = generated_network['pore.coords'][pore2][1]

                # Set up pore coordinates
                pore1_coords = [pore1_x, pore1_y]
                pore2_coords = [pore2_x, pore2_y]

                # Distance between the two pores
                distance = math.dist(pore1_coords, pore2_coords)

                # Skip if the throat diameter is not defined
                if math.isnan(
                        generated_network['throat.diameter'][throat_index]):
                    continue

                # Calculate the number of points (circles) in the throat
                # + 2 extra circles for better connectivity
                num_throat_points = math.ceil(
                    distance /
                    (generated_network['throat.diameter'][throat_index])) + 2

                # Calculate the distance between the center of each circle along the path in x and y directions,
                # and the total magnitude of the distance vector
                x_segment = (pore2_coords[0] -
                             pore1_coords[0]) / num_throat_points
                y_segment = (pore2_coords[1] -
                             pore1_coords[1]) / num_throat_points
                magnitude = generated_network['throat.diameter'][throat_index]

                # Loop to create circles representing the throat with random variations
                for i in range(num_throat_points):
                    # Calculate the base position for the current circle (along the path between pores)
                    base_point = [
                        pore1_coords[0] + (x_segment * i),
                        pore1_coords[1] + (y_segment * i)
                    ]

                    # Generate a random direction for the perpendicular vector (affects circle position)
                    direction = np.random.randint(0, 2)
                    perp_vector = [0, 0]

                    # Set the perpendicular vector based on the random direction
                    if direction == 0:
                        # Clockwise from the throat direction
                        perp_vector = [
                            y_segment / magnitude, -x_segment / magnitude
                        ]
                    else:
                        # Counter-clockwise from the throat direction
                        perp_vector = [
                            -y_segment / magnitude, x_segment / magnitude
                        ]

                    # Introduce randomness in the circle position based on throat diameter
                    throat_radius = generated_network['throat.diameter'][
                        throat_index] / 2
                    random_shift = np.random.uniform(
                        -throat_radius, throat_radius) * throat_random

                    # Apply the random shift to the perpendicular vector
                    perp_vector[0] *= random_shift
                    perp_vector[1] *= random_shift

                    # Calculate the final position of the circle (base point + offset from perpendicular vector)
                    x_new = base_point[0] + perp_vector[0]
                    y_new = base_point[1] + perp_vector[1]

                    # Add a circle to the modelspace representing the current section of the throat
                    modelspace.add_circle((x_new, y_new),
                                          radius=magnitude / 20)

                    # Create a hatch object (likely for defining fill properties) for the circle
                    hatch = modelspace.add_hatch(color=7)

                    # Add an elliptical hatch for the throat circle
                    edge_path = hatch.paths.add_edge_path()
                    edge_path.add_ellipse((x_new, y_new),
                                          major_axis=(0, magnitude / 20),
                                          ratio=1)

    return document
