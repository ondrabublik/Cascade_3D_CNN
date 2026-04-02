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
        'slice': slice,
        'tf': tf
    })

    x = np.load(os.path.join(path, "x.npy"))
    y = np.load(os.path.join(path, "y.npy"))
    z = np.load(os.path.join(path, "z.npy"))
    dataIn = np.load(os.path.join(path, "dataIn.npy"))
    dataIn = dataIn[0:1,:,:,:,:]

    nSpec, nx, ny, nz, dimIn = np.shape(dataIn)

    scales = np.load(path / Path("scales.npy"), allow_pickle=True).item()
    in_min = scales["in_min"]
    in_max = scales["in_max"]

    secondary_velocity = [-5,0,5,10,15,20,25,30,35,40,45,50,55,60] # [m/s]
    for v in secondary_velocity:
        dataIn[:, :, :, :, 3] = (v - in_min[3]) / (in_max[3] - in_min[3])

        gen = net.predict(dataIn)
        desDataIn = descaleIn(path, dataIn)

        gen = descaleOut(path, gen)
        vtk_all(pathResults / Path('result_unseen' + str(v) + '.vtu'), desDataIn[0,:,:,:,0], desDataIn[0,:,:,:,1], desDataIn[0,:,:,:,2], desDataIn[0,:,:,:,3], x, y, z, gen[0,:,:,:,0], gen[0,:,:,:,1], gen[0,:,:,:,2], gen[0,:,:,:,3])


