import numpy as np
import tensorflow as tf
import re
from tensorflow import keras
from dataClass3D import Data
from matplotlib import pyplot as plt
from pathlib import Path
import scipy
from meshDeformation3D import meshDeformation3D as meshDeformation
from toVtk import vtk, vtkBoundary


def plotResult(path, gen, dataIn, iter):

    plt.rcParams.update({'font.size': 15})

    lastX = dataIn.shape[1]
    lastY = dataIn.shape[2]
    X = dataIn[0, 0:lastX, 0:lastY, 0]
    Y = dataIn[0, 0:lastX, 0:lastY, 1]

    u = gen[0, :, :, 0]
    v = gen[0, :, :, 1]
    w = gen[0, :, :, 2]
    p = gen[0, :, :, 3]
    uv = np.sqrt(u ** 2 + v ** 2)

    velMin = np.amin(uv)
    velMax = np.amax(uv)
    pMin = np.amin(p)
    pMax = np.amax(p)
    nLevels = 20

    fig = plt.figure(figsize=(18, 16), )
    spec = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(spec[0, 0])
    plt.title('velocity', fontsize=20)
    ax0.axes.xaxis.set_visible(False)
    ax0.axes.yaxis.set_visible(False)

    ax01 = fig.add_subplot(spec[0, 1])
    plt.title('pressure', fontsize=20)
    ax01.axes.xaxis.set_visible(False)
    ax01.axes.yaxis.set_visible(False)

    ax = ax0
    ax.set_aspect('equal', adjustable='box')
    ax.contourf(X, Y, uv, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))


    ax = ax01
    ax.set_aspect('equal', adjustable='box')
    ax.contourf(X, Y, p, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))


    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(path / Path('id_' + str(iter) + '.png'), dpi=150)
    plt.close(fig)


def readMatFiles(pathDir):
    mat_files = [f for f in Path(pathDir).iterdir()]
    sorted_mat_files = sorted(mat_files, key=lambda filename: int(re.search(r'\d+', filename.name).group()))

    return sorted_mat_files


def prepareDataInFromCFD(ind, matFiles, B, dt):
    mat = scipy.io.loadmat(matFiles[ind])['data']
    nextMat = scipy.io.loadmat(matFiles[ind+1])['data']

    nx, ny, nz = np.shape(mat['X'][0][0])

    dataIn = np.zeros((1,nx,ny,nz,15))
    dataOut = np.zeros((1,nx,ny,nz,4))

    dataIn[0:1,0:nx, 0:ny, 0:nz, 0] = mat['X'][0][0]
    dataIn[0:1,0:nx, 0:ny, 0:nz, 1] = mat['Y'][0][0]
    dataIn[0:1,0:nx, 0:ny, 0:nz, 2] = mat['Z'][0][0]
    dataIn[0:1,0:nx, 0:ny, 0:nz, 3] = (nextMat['X'][0][0] - mat['X'][0][0]) / dt
    dataIn[0:1,0:nx, 0:ny, 0:nz, 4] = (nextMat['Y'][0][0] - mat['Y'][0][0]) / dt
    dataIn[0:1,0:nx, 0:ny, 0:nz, 5] = (nextMat['Z'][0][0] - mat['Z'][0][0]) / dt
    dataIn[0:1,0:nx, 0:ny, 0:nz, 6] = B
    dataIn[0:1,0:nx, 0:ny, 0:nz, 7] = mat['D_inlet'][0][0]
    dataIn[0:1,0:nx, 0:ny, 0:nz, 8] = mat['D'][0][0]
    dataIn[0:1,0:nx, 0:ny, 0:nz, 9] = mat['parameters'][0][0][0][0] / 20
    dataIn[0:1, 0:nx, 0:ny, 0:nz, 10] = mat['parameters'][0][0][0][1] / 20
    dataIn[0:1,0:nx, 0:ny, 0:nz, 11] = mat['U'][0][0]
    dataIn[0:1,0:nx, 0:ny, 0:nz, 12] = mat['V'][0][0]
    dataIn[0:1,0:nx, 0:ny, 0:nz, 13] = mat['W'][0][0]
    dataIn[0:1,0:nx, 0:ny, 0:nz, 14] = mat['P'][0][0]

    dataOut[0:1,0:nx, 0:ny, 0:nz, 0] = nextMat['U'][0][0]
    dataOut[0:1,0:nx, 0:ny, 0:nz, 1] = nextMat['V'][0][0]
    dataOut[0:1,0:nx, 0:ny, 0:nz, 2] = nextMat['W'][0][0]
    dataOut[0:1,0:nx, 0:ny, 0:nz, 3] = nextMat['P'][0][0]

    return dataIn, dataOut


def getPossitions(t, nBody):
    x = np.zeros(nBody)
    y = np.zeros(nBody)
    z = np.zeros(nBody)
    y[0] = 0.12 * np.sin(4 * t)
    # y[1] = 0.12 * np.cos(0.9 * t)
    # y[2] = 0.12 * np.sin(0.3 * t)
    return x, y, z


if __name__ == "__main__":
    dataDirs = ['../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in15_vent10']
    path = Path('../../data/net32_3D_multistep')
    pathResults = path / Path('results_CNN_multi')
    pathResults.mkdir(exist_ok=True)

    net = keras.models.load_model(path / Path("model.keras"), safe_mode=False, custom_objects={
        'slice':slice,
          'tf':tf})

    # net = keras.layers.TFSMLayer(path, call_endpoint="serving_default")
    data = Data(dataDirs)
    matFiles = readMatFiles(dataDirs[0])

    md = meshDeformation(Path(dataDirs[0]).parents[0] / Path('mesh.mat'))
    B = md.computeB()

    T = 30
    dt = 0.1

    dataCFDIn, dataCFDOut = prepareDataInFromCFD(3, matFiles=matFiles, B=B, dt=data.parameters['dt'])

    dataIn = np.zeros([1, data.nx, data.ny, data.nz, 13])

    xt, yt, zt = getPossitions(0, md.nBody)
    Xn, Yn, Zn = md.computeTiltMesh(xt, yt, zt)
    dataIn[0:1, :, :, :, 0] = Xn
    dataIn[0:1, :, :, :, 1] = Yn
    dataIn[0:1, :, :, :, 2] = Zn
    dataIn[0:1, :, :, :, 6] = md.computeB()
    dataIn[0:1, :, :, :, 7] = dataCFDIn[0:1, :, :, :, 7] * 15.0 / 20.0
    dataIn[0:1, :, :, :, 8] = dataCFDIn[0:1, :, :, :, 8] * 10.0 / 20.0
    dataIn[0:1, :, :, :, 9:13] = dataCFDIn[0:1, :, :, :, 9:13]

    for iter, t in enumerate(np.arange(dt, T, dt)):
        print("Time: " + str(round(t, 3)) + ", Iter: " + str(iter))

        xt, yt, zt = getPossitions(t, md.nBody)
        Xn1, Yn1, Zn1 = md.computeTiltMesh(xt, yt, zt)
        uMesh = (Xn1 - Xn) / dt
        vMesh = (Yn1 - Yn) / dt
        wMesh = (Zn1 - Zn) / dt
        Xn = Xn1
        Yn = Yn1
        Zn = Zn1

        dataIn[0, :, :, :, 3] = uMesh
        dataIn[0, :, :, :, 4] = vMesh
        dataIn[0, :, :, :, 5] = wMesh

        # boundary condition
        #dataIn[0, 0, :, :, 11] = 15

        gen = net.predict(dataIn)

        dataIn[0, :, :, :, 0] = Xn1
        dataIn[0, :, :, :, 1] = Yn1
        dataIn[0, :, :, :, 2] = Zn1
        dataIn[:, :, :, :, 9:13] = gen[:, :, :, :]

        plotResult(pathResults, gen[:, :, :, -5, :], dataIn[:, :, :, -5, :], iter)

        vtkBoundary(pathResults / Path('boundary_' + str(iter) + '.vtu'), B, dataIn[0, :, :, :, 0], dataIn[0, :, :, :, 1],
            dataIn[0, :, :, :, 2], gen[0, :, :, :, 3])
        vtk(pathResults / Path('result_' + str(iter) + '.vtu'), B, dataIn[0, :, :, :, 0], dataIn[0, :, :, :, 1],
            dataIn[0, :, :, :, 2], gen[0, :, :, :, 0], gen[0, :, :, :, 1], gen[0, :, :, :, 2], gen[0, :, :, :, 3])