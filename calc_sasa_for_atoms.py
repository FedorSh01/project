#! python

# this code has been copied with changes from
# https://github.com/biopython/biopython/blob/c4a47ffff7f3e7de32ab3d8846983d3531ea63b4/Bio/PDB/SASA.py#L117

import collections
import math
import numpy as np
import handmade_kdtree as my_kdt

ATOMIC_RADII = collections.defaultdict(lambda: 2.0)
ATOMIC_RADII.update(
    {
        "H": 1.200,
        "HE": 1.400,
        "C": 1.700,
        "N": 1.550,
        "O": 1.520,
        "F": 1.470,
        "NA": 2.270,
        "MG": 1.730,
        "P": 1.800,
        "S": 1.800,
        "CL": 1.750,
        "K": 2.750,
        "CA": 2.310,
        "NI": 1.630,
        "CU": 1.400,
        "ZN": 1.390,
        "SE": 1.900,
        "BR": 1.850,
        "CD": 1.580,
        "I": 1.980,
        "HG": 1.550,
    }
)


def _compute_sphere(n_points):
    """Return the 3D coordinates of n points on a sphere.
    Uses the golden spiral algorithm to place points 'evenly' on the sphere
    surface. We compute this once and then move the sphere to the centroid
    of each atom as we compute the ASAs.
    """
    n = n_points

    dl = np.pi * (3 - 5 ** 0.5)
    dz = 2.0 / n

    longitude = 0
    z = 1 - dz / 2

    coords = np.zeros((n, 3), dtype=np.float32)
    for k in range(n):
        r = (1 - z * z) ** 0.5
        coords[k, 0] = math.cos(longitude) * r
        coords[k, 1] = math.sin(longitude) * r
        coords[k, 2] = z
        z -= dz
        longitude += dl

    return coords


def compute_sasa_for_atoms(atoms: list, probe_radius: float, n_points: int, atomic_radii: dict):

    radii_dict = atomic_radii.copy()

    if probe_radius <= 0.0:
        raise ValueError(
            f"Probe radius must be a positive number: {probe_radius} <= 0"
        )

    if n_points < 1:
        raise ValueError(
            f"Number of sphere points must be larger than 1: {n_points}"
        )

    n_atoms = len(atoms)
    if not n_atoms:
        raise ValueError("where are atoms?")

    _sphere = _compute_sphere(n_points)

    # Get coordinates as a numpy array
    # We trust DisorderedAtom and friends to pick representatives.
    coords = list([a.coord for a in atoms])  # порядок как в атомах

    d_index = {}
    for f in range(len(coords)):
        d_index[tuple(coords[f])] = f  # словарь координата: индекс в массиве атомов
    # Pre-compute atom neighbors using KDTree
    kdt = my_kdt.KDtree(coords, 3)  # моё кдтри не изменяет массив координат

    # Pre-compute radius * probe table
    radii = np.array([radii_dict[a.element] for a in atoms], dtype=np.float64)  # порядок как в атомах
    radii += probe_radius
    twice_maxradii = np.max(radii) * 2

    # Calculate ASAs
    asa_array = np.zeros((n_atoms, 1), dtype=np.int64)  # пустой массив пот результат (порядок как в атомах)
    for i in range(n_atoms):

        r_i = radii[i]

        # Move sphere to atom
        s_on_i = (np.array(_sphere, copy=True) * r_i) + coords[i]
        s_on_i_my = s_on_i.tolist()

        # KDtree for sphere points
        kdt_sphere = my_kdt.KDtree(s_on_i_my, 3)

        # Iterate over neighbors of atom i
        neighbors = kdt.find_neighbors(coords[i], twice_maxradii)  # словарь узел (класс NodeKD): расстояние до него

        overlap_points = {}
        for jj in neighbors:
            if neighbors[jj] == 0:  # если расстояние до соседа = 0 (то есть это эта же точка)
                continue

            if neighbors[jj] < (r_i + radii[d_index[tuple(jj.coor)]]):
                # если расстояние до соседа < радиусов этих атомов (+ 2 мол воды)
                # Remove overlapping points on sphere from available set
                overlap_points.update(kdt_sphere.find_neighbors(jj.coor, radii[d_index[tuple(jj.coor)]]))
                # в словарь добавляем точки сферы (класс - узлы дерева), которые перекрываются с соседним атомом
                # в итоге длинна словаря - количество перекрывающихся точек со всеми соседними атомами

        asa_array[i] = n_points - len(overlap_points)  # update counts

    # Convert accessible point count to surface area in A**2
    f = radii * radii * (4 * np.pi / n_points)
    asa_array = asa_array * f[:, np.newaxis]

    # Set atom .sasa
    for i, atom in enumerate(atoms):
        atom.sasa = asa_array[i, 0]