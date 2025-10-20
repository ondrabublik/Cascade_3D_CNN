import numpy as np
import scipy.io
import os

def convert_mat_to_npy(path_to_mat):
    mat = scipy.io.loadmat(path_to_mat, squeeze_me=True, struct_as_record=False)
    data = mat['data']
    if not isinstance(data, (list, np.ndarray)):
        data = [data]

    specimens = []
    labels = []

    for A in data:
        param_value = A.parameters[0]
        param_matrix_Re = np.full_like(A.X, param_value, dtype=float)

        param_value = A.parameters[1]
        param_matrix_alfa = np.full_like(A.X, param_value, dtype=float)

        # inputs: X, Y, D + parametr
        In = np.stack([A.X, A.Y, A.B, param_matrix_Re, param_matrix_alfa], axis=-1)  # [nx, ny, 4]

        # outputs: u, v, p
        Out = np.stack([A.u, A.v, A.p], axis=-1)  # [nx, ny, 3]

        noise = np.zeros_like(Out)
        for k in range(3):
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
    in_min = dataIn.min(axis=(0,1,2), keepdims=True)
    in_max = dataIn.max(axis=(0,1,2), keepdims=True)
    out_min = dataOut.min(axis=(0,1,2), keepdims=True)
    out_max = dataOut.max(axis=(0,1,2), keepdims=True)

    dataIn_norm = (dataIn - in_min) / (in_max - in_min + 1e-12)
    dataOut_norm = (dataOut - out_min) / (out_max - out_min + 1e-12)

    folder = os.path.dirname(path_to_mat)
    np.save(os.path.join(folder, "dataIn.npy"), dataIn_norm)
    np.save(os.path.join(folder, "dataOut.npy"), dataOut_norm)

    # uložíme škálovací parametry
    scales = {
        "in_min": in_min.squeeze(),
        "in_max": in_max.squeeze(),
        "out_min": out_min.squeeze(),
        "out_max": out_max.squeeze()
    }
    np.save(os.path.join(folder, "scales.npy"), scales)

    print(f"Uloženo: dataIn.npy, dataOut.npy, scales.npy")
    print(f"Rozměry: dataIn {dataIn_norm.shape}, dataOut {dataOut_norm.shape}")

if __name__ == "__main__":
    convert_mat_to_npy('../../data/training_data/test_2D/data.mat')
