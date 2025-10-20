import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from pathlib import Path
from toVtk import vtk, vtk_all
import trainUnet3D_2 as tr


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
    dataOut = np.load(os.path.join(path, "dataOut.npy"))

    nSpec, nx, ny, nz, dimIn = np.shape(dataIn)
    nSpec, nx, ny, nz, dimOut = np.shape(dataOut)

    gen = net.predict(dataIn)
    dataIn = descaleIn(path, dataIn)
    dataOut = descaleOut(path, dataOut)
    gen = descaleOut(path, gen)
    for ind in [0,1,2,3]:
        vtk_all(pathResults / Path('result_' + str(ind) + '.vtu'), dataIn[ind,:,:,:,0], dataIn[ind,:,:,:,1],dataIn[ind,:,:,:,2],dataIn[ind,:,:,:,3], x, y, z, gen[ind,:,:,:,0], gen[ind,:,:,:,1], gen[ind,:,:,:,2], gen[ind,:,:,:,3])
        vtk(pathResults / Path('result_spec_'+ str(ind) + '.vtu'), dataIn[ind, :, :, :, 0], x, y, z,
            dataOut[ind, :, :, :, 0], dataOut[ind, :, :, :, 1], dataOut[ind, :, :, :, 2], dataOut[ind, :, :, :, 3])

