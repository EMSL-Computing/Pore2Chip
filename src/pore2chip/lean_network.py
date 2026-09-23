"""
Lightweight, dependency-free replacement for the subset of OpenPNM's network
API used by :mod:`pore2chip.generate` to build 2D pore networks.

``LeanNetwork`` mimics the parts of an OpenPNM ``Network`` object that
``generate.py`` relies on (dict-style ``'pore.*'``/``'throat.*'`` data
access, label based ``pores()``/``throats()`` lookups, and simple topology
queries) without requiring the full OpenPNM dependency for network
generation.
"""

import numpy as np
from itertools import combinations


class LeanNetwork:
    r"""
    Minimal 2D pore network container.

    Stores pore and throat data as ``'pore.<prop>'``/``'throat.<prop>'``
    dictionary entries (mirroring OpenPNM's data model) along with boolean
    label arrays that can be queried with :meth:`pores` and :meth:`throats`.
    """

    def __init__(self):
        self._data = {}
        self._pore_labels = {}
        self._throat_labels = {}
        self.params = {'dimensionality': [True, True, False]}

    # ------------------------------------------------------------------
    # Dict-like data access
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = np.asarray(value)

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __repr__(self):
        return (f"LeanNetwork(num_pores={self.Np}, num_throats={self.Nt})")

    # ------------------------------------------------------------------
    # Sizes
    # ------------------------------------------------------------------
    @property
    def Np(self):
        return len(self._data.get('pore.coords', []))

    @property
    def Nt(self):
        return len(self._data.get('throat.conns', []))

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    def set_pore_label(self, label, indices):
        mask = np.zeros(self.Np, dtype=bool)
        mask[np.asarray(indices, dtype=int)] = True
        self._pore_labels[label] = mask

    def set_throat_label(self, label, indices):
        mask = np.zeros(self.Nt, dtype=bool)
        mask[np.asarray(indices, dtype=int)] = True
        self._throat_labels[label] = mask

    def pores(self, label):
        r"""Return pore indices matching ``label``."""
        return np.where(self._pore_labels.get(label,
                                              np.zeros(self.Np, dtype=bool)))[0]

    def throats(self, label):
        r"""Return throat indices matching ``label``."""
        return np.where(self._throat_labels.get(
            label, np.zeros(self.Nt, dtype=bool)))[0]

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------
    def find_neighbor_throats(self, pores):
        r"""Return indices of throats connected to the given pore(s)."""
        pores = np.atleast_1d(pores)
        if self.Nt == 0:
            return np.array([], dtype=int)
        conns = self._data['throat.conns']
        mask = np.isin(conns[:, 0], pores) | np.isin(conns[:, 1], pores)
        return np.where(mask)[0]

    def find_neighbor_pores(self, pores, flatten=True):
        r"""Return indices of pores directly connected to the given pore(s)."""
        pores = set(np.atleast_1d(pores).tolist())
        neighbors = set()
        if self.Nt > 0:
            conns = self._data['throat.conns']
            for p1, p2 in conns:
                if p1 in pores:
                    neighbors.add(int(p2))
                if p2 in pores:
                    neighbors.add(int(p1))
        neighbors -= pores
        result = np.array(sorted(neighbors), dtype=int)
        return result

    def find_nearby_pores(self, pores, r, flatten=True):
        r"""Return indices of pores within radial distance ``r`` of the given
        pore(s), excluding the pores themselves."""
        pores = np.atleast_1d(pores)
        coords = self._data['pore.coords']
        found = set()
        for p in pores:
            dists = np.linalg.norm(coords - coords[p], axis=1)
            nearby = np.where((dists <= r) & (dists > 1e-12))[0]
            found.update(int(i) for i in nearby)
        return np.array(sorted(found), dtype=int)

    # ------------------------------------------------------------------
    # Topology modification
    # ------------------------------------------------------------------
    def add_throat(self, pore1, pore2, label=None):
        r"""Add a single throat connecting ``pore1`` and ``pore2``."""
        conns = self._data.get('throat.conns')
        new_conn = np.array([[pore1, pore2]])
        if conns is None or len(conns) == 0:
            self._data['throat.conns'] = new_conn
        else:
            self._data['throat.conns'] = np.vstack([conns, new_conn])

        for key in list(self._throat_labels.keys()):
            self._throat_labels[key] = np.append(self._throat_labels[key],
                                                 False)
        if label is not None:
            mask = np.zeros(self.Nt, dtype=bool)
            mask[-1] = True
            if label in self._throat_labels:
                self._throat_labels[label] = (self._throat_labels[label]
                                              | mask)
            else:
                self._throat_labels[label] = mask

    def trim(self, pores=None, throats=None):
        r"""Remove the given pores and/or throats from the network."""
        if pores is not None:
            pores = np.atleast_1d(pores)
            if len(pores) > 0:
                keep_pores = np.ones(self.Np, dtype=bool)
                keep_pores[pores] = False

                # Any throat touching a removed pore must go too
                conns = self._data.get('throat.conns')
                if conns is not None and len(conns) > 0:
                    keep_throats = ~(np.isin(conns[:, 0], pores)
                                    | np.isin(conns[:, 1], pores))
                    self._trim_throats(~keep_throats)

                # Remap remaining pore indices
                new_index = np.cumsum(keep_pores) - 1
                for key in self._data:
                    if key.startswith('pore.'):
                        self._data[key] = self._data[key][keep_pores]
                for key in self._pore_labels:
                    self._pore_labels[key] = self._pore_labels[key][keep_pores]

                conns = self._data.get('throat.conns')
                if conns is not None and len(conns) > 0:
                    self._data['throat.conns'] = new_index[conns]

        if throats is not None:
            throats = np.atleast_1d(throats)
            if len(throats) > 0:
                mask = np.zeros(self.Nt, dtype=bool)
                mask[throats] = True
                self._trim_throats(mask)

    def _trim_throats(self, mask):
        r"""Remove throats where ``mask`` is True."""
        keep = ~mask
        for key in list(self._data.keys()):
            if key.startswith('throat.'):
                self._data[key] = self._data[key][keep]
        for key in self._throat_labels:
            self._throat_labels[key] = self._throat_labels[key][keep]

    def connect_pores(self, pore1, pore2, labels=None):
        r"""Add a throat connecting ``pore1`` to ``pore2``."""
        label = labels[0] if labels else None
        self.add_throat(pore1, pore2, label=label)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create_2d(cls, n1, n2):
        r"""
        Build a 2D pore network laid out like an OpenPNM
        ``BodyCenteredCubic`` lattice collapsed to a single z-plane: a
        regular ``n1`` x ``n2`` grid of "corner" pores plus an
        ``(n1 - 1)`` x ``(n2 - 1)`` grid of "body" pores centered between
        every four corner pores, with each body pore connected to its four
        surrounding corner pores.

        Args:
            n1 (int): Number of corner pores along the x-axis
            n2 (int): Number of corner pores along the y-axis

        Returns:
            LeanNetwork: Network with ``pore.coords`` and ``throat.conns``
            populated (no throats between corner pores, matching the
            OpenPNM lattice after trimming ``body_to_body`` and
            ``corner_to_corner`` throats).
        """
        net = cls()

        corner_coords = []
        for j in range(n2):
            for i in range(n1):
                corner_coords.append([float(i), float(j), 0.0])
        num_corner = len(corner_coords)

        body_coords = []
        body_index = {}
        for j in range(n2 - 1):
            for i in range(n1 - 1):
                body_index[(i, j)] = num_corner + len(body_coords)
                body_coords.append([i + 0.5, j + 0.5, 0.0])

        coords = np.array(corner_coords + body_coords, dtype=float)
        net._data['pore.coords'] = coords

        conns = []
        for (i, j), body_idx in body_index.items():
            corner_indices = [
                j * n1 + i,
                j * n1 + (i + 1),
                (j + 1) * n1 + i,
                (j + 1) * n1 + (i + 1),
            ]
            for corner_idx in corner_indices:
                conns.append([corner_idx, body_idx])

        net._data['throat.conns'] = np.array(conns, dtype=int)

        # Shift and scale so pores span the full [0, n1-1] x [0, n2-1]
        # extent, matching the original OpenPNM-derived generation.
        coords[:, 0] -= 0.5
        coords[:, 1] -= 0.5
        coords[:, 0] *= (n1 / (n1 - 1))
        coords[:, 1] *= (n2 / (n2 - 1))

        return net


# ------------------------------------------------------------------------
# Functional-style helpers mirroring op.topotools / op.models.network
# ------------------------------------------------------------------------
def trim(network, pores=None, throats=None):
    r"""Remove pores and/or throats from ``network`` (mirrors
    ``openpnm.topotools.trim``)."""
    network.trim(pores=pores, throats=throats)


def connect_pores(network, pore1, pore2, labels=None):
    r"""Connect two pores with a new throat (mirrors
    ``openpnm.topotools.connect_pores``)."""
    network.connect_pores(pore1, pore2, labels=labels)


def duplicate_throats(network):
    r"""Return indices of throats that duplicate an earlier throat's
    connection (mirrors ``openpnm.models.network.duplicate_throats``)."""
    if network.Nt == 0:
        return np.array([], dtype=int)
    conns = network['throat.conns']
    seen = set()
    dupes = []
    for idx, (p1, p2) in enumerate(conns):
        key = (min(p1, p2), max(p1, p2))
        if key in seen:
            dupes.append(idx)
        else:
            seen.add(key)
    return np.array(dupes, dtype=int)


def coordination_number(network):
    r"""Return the number of throats connected to each pore (mirrors
    ``openpnm.models.network.coordination_number``)."""
    coord = np.zeros(network.Np, dtype=int)
    if network.Nt > 0:
        conns = network['throat.conns']
        for p1, p2 in conns:
            coord[p1] += 1
            coord[p2] += 1
    return coord


def reduce_coordination(network, average_coord):
    r"""Return indices of throats to trim so the network's average
    coordination number is reduced to approximately ``average_coord``
    (mirrors ``openpnm.topotools.reduce_coordination``).

    Throats are removed one at a time (favoring throats attached to the
    currently most-connected pores) while making sure no pore is left
    completely isolated.
    """
    if network.Nt == 0:
        return np.array([], dtype=int)

    conns = network['throat.conns'].copy()
    coord = coordination_number(network)
    target_total = average_coord * network.Np
    to_remove = []
    remaining = list(range(network.Nt))

    while remaining and coord.sum() > target_total:
        remaining.sort(key=lambda t: -max(coord[conns[t][0]], coord[
            conns[t][1]]))
        throat = remaining[0]
        p1, p2 = conns[throat]
        if coord[p1] <= 1 or coord[p2] <= 1:
            remaining.remove(throat)
            continue
        to_remove.append(throat)
        remaining.remove(throat)
        coord[p1] -= 1
        coord[p2] -= 1

    return np.array(to_remove, dtype=int)
