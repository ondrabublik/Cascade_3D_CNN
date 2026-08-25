import numpy as np
import pyvista as pv
from vtk import VTK_HEXAHEDRON, VTK_QUAD

# Cache podle shape pole X
_vtk_cell_cache = {}

def vtk(filename, B, X, Y, Z, U, V, W, P):
    nx, ny, nz = np.shape(X)

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    boundary = B.flatten()

    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    # Generuj seznam buněk (každý hexahedron má 8 bodů)
    if "vtk" in _vtk_cell_cache:
        cells, cell_types = _vtk_cell_cache["vtk"]
    else:
        cells = []
        cell_types = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    p0 = idx(i, j, k)
                    p1 = idx(i + 1, j, k)
                    p2 = idx(i + 1, j + 1, k)
                    p3 = idx(i, j + 1, k)
                    p4 = idx(i, j, k + 1)
                    p5 = idx(i + 1, j, k + 1)
                    p6 = idx(i + 1, j + 1, k + 1)
                    p7 = idx(i, j + 1, k + 1)

                    # VTK očekává: počet bodů + indexy bodů
                    if np.sum(boundary[[p0, p1, p2, p3, p4, p5, p6, p7]]) < 8:
                        cells.append([8, p0, p1, p2, p3, p4, p5, p6, p7])
                        cell_types.append(VTK_HEXAHEDRON)

        cells = np.array(cells, dtype=np.int64).flatten()
        _vtk_cell_cache["vtk"] = (cells, cell_types)

    # create unstructuredGrid
    ugrid = pv.UnstructuredGrid(cells, cell_types, points)

    velocity = np.stack([U.flatten(), V.flatten(), W.flatten()], axis=1)
    pressure = P.flatten()

    ugrid.point_data['velocity'] = velocity
    ugrid.point_data['pressure'] = pressure

    # Ulož do souboru
    ugrid.save(filename)
    print("File saved: " + str(filename))


def vtkBoundary(filename, B, X, Y, Z, P, bladeIndex=None):
    """Export pressure on blade surface only (quads where B==1 faces fluid cells)."""
    nx, ny, nz = np.shape(X)
    points_all = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    pressure_all = np.asarray(P).reshape(-1)
    blade_index_all = None if bladeIndex is None else np.asarray(bladeIndex).reshape(-1)
    boundary = B.flatten()

    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    cache_key = ("boundary", nx, ny, nz)
    if cache_key in _vtk_cell_cache:
        used, cells, cell_types = _vtk_cell_cache[cache_key]
    else:
        faces = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    p0 = idx(i, j, k)
                    p1 = idx(i + 1, j, k)
                    p2 = idx(i + 1, j + 1, k)
                    p3 = idx(i, j + 1, k)
                    p4 = idx(i, j, k + 1)
                    p5 = idx(i + 1, j, k + 1)
                    p6 = idx(i + 1, j + 1, k + 1)
                    p7 = idx(i, j + 1, k + 1)

                    # Fluid cell adjacent to solid (blade) face
                    if np.sum(boundary[[p0, p1, p2, p3, p4, p5, p6, p7]]) >= 8:
                        continue

                    candidates = (
                        (p0, p1, p2, p3),
                        (p4, p5, p6, p7),
                        (p0, p1, p5, p4),
                        (p1, p2, p6, p5),
                        (p2, p3, p7, p6),
                        (p3, p0, p4, p7),
                    )
                    for face in candidates:
                        if np.sum(boundary[list(face)]) == 4:
                            faces.append(face)

        if not faces:
            raise ValueError("No blade-surface quads found (check B mask).")

        used = np.unique(np.asarray(faces, dtype=np.int64).ravel())
        remap = {int(old): new for new, old in enumerate(used)}

        cells = []
        cell_types = []
        for face in faces:
            cells.extend([4, remap[face[0]], remap[face[1]], remap[face[2]], remap[face[3]]])
            cell_types.append(VTK_QUAD)

        cells = np.asarray(cells, dtype=np.int64)
        cell_types = np.asarray(cell_types, dtype=np.uint8)
        _vtk_cell_cache[cache_key] = (used, cells, cell_types)

    ugrid = pv.UnstructuredGrid(cells, cell_types, points_all[used])
    ugrid.point_data['pressure'] = pressure_all[used]
    if blade_index_all is not None:
        ugrid.point_data['blade_index'] = blade_index_all[used]

    ugrid.save(filename)
    print("File saved: " + str(filename))
