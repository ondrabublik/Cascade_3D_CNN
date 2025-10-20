import numpy as np
import os


def readData(path):
    # Načtení souřadnic a typu bodu
    print(path + '/vertices.txt')
    vertices = np.loadtxt(path + '/vertices.txt')  # shape: (n_points, 4)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    b = vertices[:, 3]

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
    data_points = np.column_stack((x, y, z, b, np.full_like(x, parameter), u, v, w, p))
    print("Shape data_points:", data_points.shape)

    nx, ny, nz = 160, 128, 64
    dim = data_points.shape[1]

    # Převod na 4D pole
    data_4d = data_points.reshape((nx, ny, nz, dim), order='F')
    print("Shape data_4d:", data_4d.shape)

    return data_4d

def convert_txt_data_to_npy(base_dir):
    specimens = []
    labels = []
    for item in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, item)

        # Jenom adresáře
        if os.path.isdir(subdir_path) and item != "results":
            print(subdir_path)
            data = readData(subdir_path)

            # inputs: X, Y, D + parametr
            In = data[:, :, :, 0:5]
            Out = data[:, :, :, 5:9]

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
    c = 0.1
    in_min = dataIn.min(axis=(0,1,2,3), keepdims=True)*(1-c)
    in_max = dataIn.max(axis=(0,1,2,3), keepdims=True)*(1+c)
    out_min = dataOut.min(axis=(0,1,2,3), keepdims=True)*(1-c)
    out_max = dataOut.max(axis=(0,1,2,3), keepdims=True)*(1+c)

    dataIn_norm = (dataIn - in_min) / (in_max - in_min + 1e-12)
    dataOut_norm = (dataOut - out_min) / (out_max - out_min + 1e-12)

    np.save(os.path.join(base_dir, "dataIn.npy"), dataIn_norm)
    np.save(os.path.join(base_dir, "dataOut.npy"), dataOut_norm)

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