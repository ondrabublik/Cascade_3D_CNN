import numpy as np
import tensorflow as tf
import re
from tensorflow import keras
from dataClass3D import Data
from pathlib import Path
import scipy
from meshDeformation3D import meshDeformation3D as meshDeformation
from toVtk import vtkBoundary


def readMatFiles(pathDir):
    md = meshDeformation(Path(pathDir).parents[0] / Path('mesh.mat'))
    B = md.computeB()
    bladeIndex = md.computeBladeIndex()

    mat_files = [f for f in Path(pathDir).iterdir()]
    sorted_mat_files = sorted(mat_files, key=lambda filename: int(re.search(r'\d+', filename.name).group()))

    return B, bladeIndex, sorted_mat_files


def prepareDataInFromCFD(ind, matFiles, B, dt):
    mat = scipy.io.loadmat(matFiles[ind])['data']
    nextMat = scipy.io.loadmat(matFiles[ind + 1])['data']

    nx, ny, nz = np.shape(mat['X'][0][0])

    dataIn = np.zeros((1, nx, ny, nz, 13))
    dataOut = np.zeros((1, nx, ny, nz, 4))

    dataIn[0:1, 0:nx, 0:ny, 0:nz, 0] = mat['X'][0][0]
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 1] = mat['Y'][0][0]
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 2] = mat['Z'][0][0]
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 3] = (nextMat['X'][0][0] - mat['X'][0][0]) / dt
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 4] = (nextMat['Y'][0][0] - mat['Y'][0][0]) / dt
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 5] = (nextMat['Z'][0][0] - mat['Z'][0][0]) / dt
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 6] = B
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 7] = mat['D'][0][0]
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 8] = mat['parameters'][0][0][0][0]
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 9] = mat['U'][0][0]
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 10] = mat['V'][0][0]
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 11] = mat['W'][0][0]
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 12] = mat['P'][0][0]

    dataOut[0:1, 0:nx, 0:ny, 0:nz, 0] = nextMat['U'][0][0]
    dataOut[0:1, 0:nx, 0:ny, 0:nz, 1] = nextMat['V'][0][0]
    dataOut[0:1, 0:nx, 0:ny, 0:nz, 2] = nextMat['W'][0][0]
    dataOut[0:1, 0:nx, 0:ny, 0:nz, 3] = nextMat['P'][0][0]

    return dataIn, dataOut, nextMat['X'][0][0], nextMat['Y'][0][0], nextMat['Z'][0][0]


if __name__ == "__main__":
    projectDir = Path(__file__).resolve().parents[1]

    dataDirs = [
        str(projectDir.parent / "reader3D" / "FinalBladeCascade" / "data" / "transformed_15o")
    ]

    path = projectDir / "data" / "net7_3D_multistep_lowo"
    pathResults = path / Path('results_blade_pressure_vent15')
    pathResults.mkdir(exist_ok=True)

    net = keras.models.load_model(path / Path("model_best.keras"), safe_mode=False, custom_objects={
        'slice': slice,
        'tf': tf})

    data = Data(dataDirs)
    B, bladeIndex, matFiles = readMatFiles(dataDirs[0])

    dataNN, _, _, _, _ = prepareDataInFromCFD(0, matFiles=matFiles, B=B, dt=data.parameters['dt'])
    for ind in range(len(matFiles) - 1):
        print(str(ind) + " / " + str(len(matFiles) - 1))
        dataIn, dataOut, Xf, Yf, Zf = prepareDataInFromCFD(ind, matFiles=matFiles, B=B, dt=data.parameters['dt'])

        dataNN[:, :, :, :, 0:9] = dataIn[:, :, :, :, 0:9]
        gen = net.predict(dataNN)
        dataNN[:, :, :, :, 9:13] = gen[:, :, :, :]

        p_unet = gen[0, :, :, :, 3]
        p_cfd = dataOut[0, :, :, :, 3]

        vtkBoundary(pathResults / Path('pressure_UNet_' + str(ind) + '.vtu'), B, Xf, Yf, Zf, p_unet, bladeIndex)
        vtkBoundary(pathResults / Path('pressure_CFD_' + str(ind) + '.vtu'), B, Xf, Yf, Zf, p_cfd, bladeIndex)
        vtkBoundary(pathResults / Path('pressure_diff_' + str(ind) + '.vtu'), B, Xf, Yf, Zf, p_unet - p_cfd, bladeIndex)
