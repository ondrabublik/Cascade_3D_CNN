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


def vtkBoundary(filename, B, X, Y, Z, P):
    nx, ny, nz = np.shape(X)

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    boundary = B.flatten()

    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    # Generuj seznam buněk
    if "boundary" in _vtk_cell_cache:
        cells, cell_types = _vtk_cell_cache["boundary"]
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
                        if np.sum(boundary[[p0, p1, p2, p3]]) == 4:
                            cells.append([4, p0, p1, p2, p3])
                            cell_types.append(VTK_QUAD)
                        if np.sum(boundary[[p4, p5, p6, p7]]) == 4:
                            cells.append([4, p4, p5, p6, p7])
                            cell_types.append(VTK_QUAD)
                        if np.sum(boundary[[p0, p1, p5, p4]]) == 4:
                            cells.append([4, p0, p1, p5, p4])
                            cell_types.append(VTK_QUAD)
                        if np.sum(boundary[[p1, p2, p6, p5]]) == 4:
                            cells.append([4, p1, p2, p6, p5])
                            cell_types.append(VTK_QUAD)
                        if np.sum(boundary[[p2, p3, p7, p6]]) == 4:
                            cells.append([4, p2, p3, p7, p6])
                            cell_types.append(VTK_QUAD)
                        if np.sum(boundary[[p3, p0, p4, p7]]) == 4:
                            cells.append([4, p3, p0, p4, p7])
                            cell_types.append(VTK_QUAD)

        cells = np.array(cells, dtype=np.int64).flatten()
        _vtk_cell_cache["boundary"] = (cells, cell_types)

    # create unstructuredGrid
    ugrid = pv.UnstructuredGrid(cells, cell_types, points)

    pressure = P.flatten()
    ugrid.point_data['pressure'] = pressure

    # Ulož do souboru
    ugrid.save(filename)
    print("File saved: " + str(filename))