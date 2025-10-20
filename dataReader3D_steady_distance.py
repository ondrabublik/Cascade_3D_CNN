import numpy as np
import os
from scipy.spatial import cKDTree

def distance(x, y, z, b, level):
    # Spojení souřadnic do jednoho pole Nx3
    points = np.column_stack((x, y, z))

    # Body s daným levelem
    mask_level = (np.abs(b-level) < 0.1)
    points_level = points[mask_level]

    # Inicializace výstupu
    distances = np.zeros(len(points))

    # Pokud neexistuje žádný bod s daným levelem
    if len(points_level) == 0:
        distances[:] = np.nan  # nebo libovolná jiná konvence
        return distances

    # Vytvoření KD-tree z bodů s daným levelem
    tree = cKDTree(points_level)

    # Body mimo level
    mask_other = ~mask_level
    points_other = points[mask_other]

    # Výpočet nejbližší vzdálenosti
    dists, _ = tree.query(points_other)

    # Uložení vzdáleností
    distances[mask_other] = dists
    distances[mask_level] = 0.0  # vzdálenost k sobě = 0

    return distances


def readData(path):
    # Načtení souřadnic a typu bodu
    print(path + '/vertices.txt')
    vertices = np.loadtxt(path + '/vertices.txt')  # shape: (n_points, 4)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    b = vertices[:, 3]

    distance_wall = distance(x, y, z, b, 1)
    distance_secondary_inlet = distance(x, y, z, b, 2)
    distance_secondary_outlet = distance(x, y, z, b, 3)

    parameter = np.loadtxt(path + '/parameters.txt')

    # Načtení dalších polí
    u = np.loadtxt(path + '/u.txt')
    v = np.loadtxt(path + '/v.txt')
    w = np.loadtxt(path + '/w.txt')
    p = np.loadtxt(path + '/p.txt')

    for i in range(len(b)):
        if b[i] == 1:
            u[i] = 0
            v[i] = 0
            w[i] = 0

    # Spojení všech dat do jednoho pole (n_points, dim)
    data_points = np.column_stack((distance_wall, distance_secondary_inlet, distance_secondary_outlet, np.full_like(x, parameter), u, v, w, p))
    print("Shape data_points:", data_points.shape)

    nx, ny, nz = 160, 128, 64
    dim = data_points.shape[1]

    # Převod na 4D pole
    data_4d = data_points.reshape((nx, ny, nz, dim), order='F')
    print("Shape data_4d:", data_4d.shape)

    return data_4d, x.reshape((nx, ny, nz), order='F'), y.reshape((nx, ny, nz), order='F'), z.reshape((nx, ny, nz), order='F')

def convert_txt_data_to_npy(base_dir):
    specimens = []
    labels = []
    x = []
    y = []
    z = []
    for item in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, item)

        # Jenom adresáře
        if os.path.isdir(subdir_path) and item != "results" and item != "models":
            print(subdir_path)
            data, x, y, z = readData(subdir_path)

            # inputs: X, Y, D + parametr
            In = data[:, :, :, 0:4]
            Out = data[:, :, :, 4:8]

            noise = np.zeros_like(Out)
            for k in range(4):
                scale = 0.01 * np.max(np.abs(Out[..., k]))  # 1 % z max hodnoty každé veličiny
                noise[..., k] = np.random.normal(0.0, scale, size=Out[..., k].shape)

            # add noise
            Out += noise

            specimens.append(In)
            labels.append(Out)

    dataIn = np.array(specimens, dtype=float)   # [nSpecimen, nx, ny, 4]
    dataOut = np.array(labels, dtype=float)     # [nSpecimen, nx, ny, 3]

    # --- Normalizace ---
    # pro každý kanál zvlášť
    in_min = dataIn.min(axis=(0,1,2,3), keepdims=True)
    in_max = dataIn.max(axis=(0,1,2,3), keepdims=True)
    out_min = dataOut.min(axis=(0,1,2,3), keepdims=True)
    out_max = dataOut.max(axis=(0,1,2,3), keepdims=True)

    dataIn_norm = (dataIn - in_min) / (in_max - in_min + 1e-12)
    dataOut_norm = (dataOut - out_min) / (out_max - out_min + 1e-12)

    np.save(os.path.join(base_dir, "dataIn.npy"), dataIn_norm)
    np.save(os.path.join(base_dir, "dataOut.npy"), dataOut_norm)
    np.save(os.path.join(base_dir, "x.npy"), x)
    np.save(os.path.join(base_dir, "y.npy"), y)
    np.save(os.path.join(base_dir, "z.npy"), z)

    # uložíme škálovací parametry
    scales = {
        "in_min": in_min.squeeze(),
        "in_max": in_max.squeeze(),
        "out_min": out_min.squeeze(),
        "out_max": out_max.squeeze()
    }
    np.save(os.path.join(base_dir, "scales.npy"), scales)

    print(f"Uloženo: dataIn.npy, dataOut.npy, scales.npy")
    print(f"Rozměry: dataIn {dataIn_norm.shape}, dataOut {dataOut_norm.shape}")

if __name__ == "__main__":
    convert_txt_data_to_npy('../data/training_data/test_3D')