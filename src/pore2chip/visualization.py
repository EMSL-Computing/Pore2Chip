"""
Simple 2D matplotlib-based visualization helpers for
:class:`pore2chip.lean_network.LeanNetwork`, mirroring the interface of
``openpnm.visualization.plot_connections`` and
``openpnm.visualization.plot_coordinates``.
"""

import numpy as np


def plot_connections(network,
                     throats=None,
                     ax=None,
                     size_by=None,
                     color_by=None,
                     cmap='turbo',
                     color='b',
                     alpha=1.0,
                     linestyle='solid',
                     linewidth=1,
                     **kwargs):
    r"""
    Produce a 2D plot showing throats as lines connecting pore coordinates.

    Args:
        network (LeanNetwork): The network whose throats to plot.
        throats (array_like, optional): Indices of the throats to plot. If
            not given, all throats are shown.
        ax (matplotlib.axes.Axes, optional): Axis to plot onto. If not
            given, a new figure and axis are created. Passing an existing
            ``ax`` allows overlaying with :func:`plot_coordinates`.
        size_by (array_like, optional): Throat values used to scale
            ``linewidth`` for each throat.
        color_by (array_like, optional): Throat values used to color each
            line according to ``cmap``.
        cmap (str): Matplotlib colormap used when ``color_by`` is given.
        color (str): Matplotlib color used when ``color_by`` is not given.
        alpha (float): Line transparency (1 is solid, 0 is invisible).
        linestyle (str): Matplotlib linestyle for the throats.
        linewidth (float): Line width, or scaling factor for ``size_by``.
        **kwargs: Additional keyword arguments passed to
            ``matplotlib.collections.LineCollection``.

    Returns:
        matplotlib.collections.LineCollection: The plotted throat lines, or
        ``None`` if the network has no throats.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        fig, ax = plt.subplots()

    Ts = np.arange(network.Nt) if throats is None else np.atleast_1d(throats)

    if len(Ts) == 0:
        return None

    coords = network['pore.coords']
    conns = network['throat.conns'][Ts]
    segments = coords[conns][:, :, :2]

    linewidths = linewidth
    if size_by is not None:
        size_by = np.asarray(size_by)[Ts]
        linewidths = linewidth * size_by / size_by.max()

    colors = color
    array = None
    if color_by is not None:
        array = np.asarray(color_by)[Ts]
        colors = None

    lc = LineCollection(segments,
                        color=colors,
                        cmap=cmap,
                        alpha=alpha,
                        linestyle=linestyle,
                        linewidths=linewidths,
                        **kwargs)
    if array is not None:
        lc.set_array(array)

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')

    return lc


def plot_coordinates(network,
                     pores=None,
                     ax=None,
                     size_by=None,
                     color_by=None,
                     label_by=None,
                     cmap='turbo',
                     color='r',
                     alpha=1.0,
                     marker='o',
                     markersize=10,
                     **kwargs):
    r"""
    Produce a 2D plot showing specified pore coordinates as markers.

    Args:
        network (LeanNetwork): The network whose pore coordinates to plot.
        pores (array_like, optional): Indices of the pores to plot. If not
            given, all pores are shown.
        ax (matplotlib.axes.Axes, optional): Axis to plot onto. If not
            given, a new figure and axis are created. Passing an existing
            ``ax`` allows overlaying with :func:`plot_connections`.
        size_by (array_like, optional): Pore values used to scale
            ``markersize`` for each pore (controls marker area).
        color_by (array_like, optional): Pore values used to color each
            marker according to ``cmap``.
        label_by (array_like, optional): Values used to label each pore
            (drawn as text next to the marker).
        cmap (str): Matplotlib colormap used when ``color_by`` is given.
        color (str): Matplotlib color used when ``color_by`` is not given.
        alpha (float): Marker transparency (1 is solid, 0 is invisible).
        marker (str): Matplotlib marker style.
        markersize (float): Marker size, or scaling factor for ``size_by``.
        **kwargs: Additional keyword arguments passed to
            ``matplotlib.axes.Axes.scatter``.

    Returns:
        matplotlib.collections.PathCollection: The plotted pore markers, or
        ``None`` if no pores were selected.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    Ps = np.arange(network.Np) if pores is None else np.atleast_1d(pores)

    if len(Ps) == 0:
        return None

    coords = network['pore.coords'][Ps]

    sizes = markersize
    if size_by is not None:
        size_by = np.asarray(size_by)[Ps]
        sizes = markersize * size_by / size_by.max()

    colors = color
    array = None
    if color_by is not None:
        array = np.asarray(color_by)[Ps]
        colors = array

    pc = ax.scatter(coords[:, 0],
                    coords[:, 1],
                    c=colors,
                    cmap=cmap if array is not None else None,
                    alpha=alpha,
                    marker=marker,
                    s=sizes,
                    **kwargs)

    if label_by is not None:
        label_by = np.atleast_1d(label_by)
        for i, pore_index in enumerate(Ps):
            ax.annotate(str(label_by[i]), (coords[i, 0], coords[i, 1]))

    ax.set_aspect('equal')

    return pc
