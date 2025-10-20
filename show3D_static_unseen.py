import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from pathlib import Path
from toVtk import vtk, vtk_all


def descaleIn(path, dataIn_norm):
    # načtení uložených škál
    scales = np.load(path / Path("scales.npy"), allow_pickle=True).item()

    in_min = scales["in_min"]
    in_max = scales["in_max"]

    # descaling
    dataIn_descale = dataIn_norm * (in_max - in_min) + in_min

    return dataIn_descale


def descaleOut(path, dataOut_norm):
    # načtení uložených škál
    scales = np.load(path / Path("scales.npy"), allow_pickle=True).item()

    out_min = scales["out_min"]
    out_max = scales["out_max"]

    # descaling
    dataOut_descale = dataOut_norm * (out_max - out_min) + out_min

    return dataOut_descale


if __name__ == "__main__":
    dataDirs = ['../data/training_data/test_3D']
    path = Path('../data/training_data/test_3D')
    pathResults = path / Path('results')
    pathResults.mkdir(exist_ok=True)

    net = keras.models.load_model(path / Path("model.keras"), safe_mode=False, compile=False, custom_objects={
        'slice':slice,
          'tf':tf
    })

    x = np.load(os.path.join(path, "x.npy"))
    y = np.load(os.path.join(path, "y.npy"))
    z = np.load(os.path.join(path, "z.npy"))
    dataIn = np.load(os.path.join(path, "dataIn.npy"))
    dataIn = dataIn[0:1,:,:,:,:]
    dataOut = np.load(os.path.join(path, "dataOut.npy"))
    dataOut = dataOut[0:1, :, :, :, :]

    nSpec, nx, ny, nz, dimIn = np.shape(dataIn)
    nSpec, nx, ny, nz, dimOut = np.shape(dataOut)
    print(nSpec)

    scales = np.load(path / Path("scales.npy"), allow_pickle=True).item()
    in_min = scales["in_min"]
    in_max = scales["in_max"]
    secondary_velocity = 10 # [m/s]
    dataIn[0:1, :, :, :, 3] = (secondary_velocity - in_min[3]) / (in_max[3] - in_min[3])

    gen = net.predict(dataIn)
    dataIn = descaleIn(path, dataIn)
    dataOut = descaleOut(path, dataOut)

    gen = descaleOut(path, gen)
    vtk_all(pathResults / Path('result_unseen.vtu'), dataIn[0,:,:,:,0], dataIn[0,:,:,:,1],dataIn[0,:,:,:,2],dataIn[0,:,:,:,3], x, y, z, gen[0,:,:,:,0], gen[0,:,:,:,1], gen[0,:,:,:,2], gen[0,:,:,:,3])


