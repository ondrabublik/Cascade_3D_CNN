import numpy as np
import tensorflow as tf
import re
from tensorflow import keras
from dataClass3D import Data
from matplotlib import pyplot as plt
from pathlib import Path
import scipy
from meshDeformation3D import meshDeformation3D as meshDeformation
from toVtk import vtk
from UNetDev3D_v2 import ReflectPadding3D


def plotResult(path, gen, dataIn, dataOut, ind):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams.update({'font.size': 15})

    lastX = dataIn.shape[1]
    lastY = dataIn.shape[2]
    X = dataIn[0, 0:lastX, 0:lastY, 0]
    Y = dataIn[0, 0:lastX, 0:lastY, 1]
    Z = dataIn[0, 0:lastX, 0:lastY, 2]
    uMesh = dataIn[0, 0:lastX, 0:lastY, 3]
    vMesh = dataIn[0, 0:lastX, 0:lastY, 4]
    zMesh = dataIn[0, 0:lastX, 0:lastY, 5]
    B = dataIn[0, 0:lastX, 0:lastY, 6]

    u = gen[0, :, :, 0]
    v = gen[0, :, :, 1]
    w = gen[0, :, :, 2]
    p = gen[0, :, :, 3]
    p[B==1]=0
    uv = np.sqrt(u ** 2 + v ** 2)
    uv[B == 1] = 0
    u0 = dataOut[0, :, :, 0]
    v0 = dataOut[0, :, :, 1]
    w0 = dataOut[0, :, :, 2]
    p0 = dataOut[0, :, :, 3]
    p0[B == 1] = 0
    uv0 = np.sqrt(u0 ** 2 + v0 ** 2)
    uv0[B == 1] = 0

    velMin = np.amin(uv0) #min(np.amin(uv), np.amin(uv0))
    velMax = np.amax(uv0) #max(np.amax(uv), np.amax(uv0))
    pMin = np.amin(p) #min(np.amin(p), np.amin(p0))
    pMax = np.amax(p) #max(np.amax(p), np.amax(p0))
    dXmin = min(np.amin(uMesh), np.amin(vMesh))
    dXMax = max(np.amax(uMesh), np.amax(vMesh))
    nLevels = 20

    fig = plt.figure(figsize=(18, 16), )
    spec = fig.add_gridspec(3, 3)
    ax0 = fig.add_subplot(spec[0, 0])
    plt.title('DNN - Velocity', fontsize=20)
    ax0.axes.xaxis.set_visible(False)
    ax0.axes.yaxis.set_visible(False)

    ax01 = fig.add_subplot(spec[0, 1])
    plt.title('CFD - Velocity', fontsize=20)
    ax01.axes.xaxis.set_visible(False)
    ax01.axes.yaxis.set_visible(False)

    ax02 = fig.add_subplot(spec[0, 2])
    plt.title('Absolute error - Velocity', fontsize=20)
    ax02.axes.xaxis.set_visible(False)
    ax02.axes.yaxis.set_visible(False)

    ax10 = fig.add_subplot(spec[1, 0])
    plt.title('DNN - Pressure', fontsize=20)
    ax10.axes.xaxis.set_visible(False)
    ax10.axes.yaxis.set_visible(False)

    ax11 = fig.add_subplot(spec[1, 1])
    plt.title('CFD - Pressure', fontsize=20)
    ax11.axes.xaxis.set_visible(False)
    ax11.axes.yaxis.set_visible(False)

    ax12 = fig.add_subplot(spec[1, 2])
    plt.title('Absolute error - Pressure', fontsize=20)
    ax12.axes.xaxis.set_visible(False)
    ax12.axes.yaxis.set_visible(False)

    ax20 = fig.add_subplot(spec[2, 0])
    plt.title('Mesh', fontsize=20)
    ax20.axes.xaxis.set_visible(False)
    ax20.axes.yaxis.set_visible(False)

    ax21 = fig.add_subplot(spec[2, 1])
    plt.title('normalized Dx', fontsize=20)
    ax21.axes.xaxis.set_visible(False)
    ax21.axes.yaxis.set_visible(False)

    ax22 = fig.add_subplot(spec[2, 2])
    plt.title('normalized Dy', fontsize=20)
    ax22.axes.xaxis.set_visible(False)
    ax22.axes.yaxis.set_visible(False)

    ax = ax0
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.contourf(X, Y, uv, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))

    ax = ax01
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, uv0, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    ax = ax02
    errVel = abs(uv - uv0)
    maxErrVel = np.amax(errVel)
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, errVel, 20, cmap='jet', levels=np.linspace(0, maxErrVel, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    ax = ax10
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.contourf(X, Y, p, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))

    ax = ax11
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, p0, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    ax = ax12
    errP = abs(p - p0)  # / p0 * 100
    maxErrP = np.amax(errP)
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, errP, 20, cmap='jet', levels=np.linspace(0, maxErrP, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    ax = ax20
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    # ax.contourf(X, Y, B, cmap='jet', alpha=0.2)
    mask = B == 0
    ax.pcolor(X, Y, B, cmap='viridis', alpha=1.0 - mask.astype(float))
    for i in range(len(X[:, 1])):
        ax.plot(X[i, :], Y[i, :], color='k', linewidth=0.3)
    for j in range(len(X[1, :])):
        ax.plot(X[:, j], Y[:, j], color='k', linewidth=0.3)

    ax = ax21
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, uMesh, 20, cmap='jet', levels=np.linspace(dXmin, dXMax, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.5f')

    ax = ax22
    # ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, vMesh, 20, cmap='jet', levels=np.linspace(dXmin, dXMax, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.5f')

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(path / Path('id_' + str(ind) + '.png'), dpi=150)
    plt.close(fig)


def readMatFiles(pathDir):
    md = meshDeformation(Path(pathDir).parents[0] / Path('mesh.mat'))
    B = md.computeB()

    mat_files = [f for f in Path(pathDir).iterdir()]
    sorted_mat_files = sorted(mat_files, key=lambda filename: int(re.search(r'\d+', filename.name).group()))

    return B, sorted_mat_files


def prepareDataInFromCFD(ind, matFiles, B, dt):
    mat = scipy.io.loadmat(matFiles[ind])['data']
    nextMat = scipy.io.loadmat(matFiles[ind+1])['data']

    nx, ny, nz = np.shape(mat['X'][0][0])

    dataIn = np.zeros((1,nx,ny,nz,13))
    dataOut = np.zeros((1,nx,ny,nz,4))

    # dataIn[0:1,0:nx, 0:ny, 0:nz, 0] = mat['X'][0][0]
    # dataIn[0:1,0:nx, 0:ny, 0:nz, 1] = mat['Y'][0][0]
    # dataIn[0:1,0:nx, 0:ny, 0:nz, 2] = mat['Z'][0][0]
    # dataIn[0:1,0:nx, 0:ny, 0:nz, 3] = (nextMat['X'][0][0] - mat['X'][0][0]) / dt
    # dataIn[0:1,0:nx, 0:ny, 0:nz, 4] = (nextMat['Y'][0][0] - mat['Y'][0][0]) / dt
    # dataIn[0:1,0:nx, 0:ny, 0:nz, 5] = (nextMat['Z'][0][0] - mat['Z'][0][0]) / dt
    # dataIn[0:1, 0:nx, 0:ny, 0:nz, 6] = B
    # dataIn[0:1, 0:nx, 0:ny, 0:nz, 7] = mat['D_inlet'][0][0]
    # dataIn[0:1, 0:nx, 0:ny, 0:nz, 8] = mat['D'][0][0]
    # dataIn[0:1, 0:nx, 0:ny, 0:nz, 9] = mat['parameters'][0][0][0][0] / 20
    # dataIn[0:1, 0:nx, 0:ny, 0:nz, 10] = mat['parameters'][0][0][0][1] / 20
    # dataIn[0:1,0:nx, 0:ny, 0:nz, 11] = mat['U'][0][0]
    # dataIn[0:1,0:nx, 0:ny, 0:nz, 12] = mat['V'][0][0]
    # dataIn[0:1,0:nx, 0:ny, 0:nz, 13] = mat['W'][0][0]
    # dataIn[0:1,0:nx, 0:ny, 0:nz, 14] = mat['P'][0][0]

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

    dataOut[0:1,0:nx, 0:ny, 0:nz, 0] = nextMat['U'][0][0]
    dataOut[0:1,0:nx, 0:ny, 0:nz, 1] = nextMat['V'][0][0]
    dataOut[0:1,0:nx, 0:ny, 0:nz, 2] = nextMat['W'][0][0]
    dataOut[0:1,0:nx, 0:ny, 0:nz, 3] = nextMat['P'][0][0]

    return dataIn, dataOut, nextMat['X'][0][0], nextMat['Y'][0][0], nextMat['Z'][0][0]


if __name__ == "__main__":
    dataDirs = ['../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent15']
    path = Path('../../data/net41_3D_multistep')
    pathResults = path / Path('results_NN_vs_CFD_in10_vent15')
    pathResults.mkdir(exist_ok=True)

    net = keras.models.load_model(path / Path("model.keras"), safe_mode=False, custom_objects={
        'slice':slice,
          'tf':tf})

    # net = keras.layers.TFSMLayer(path, call_endpoint="serving_default")
    data = Data(dataDirs)
    B, matFiles = readMatFiles(dataDirs[0])

    dataNN, dataOut0, Xf, Yf, Zf = prepareDataInFromCFD(0, matFiles=matFiles, B=B, dt=data.parameters['dt'])
    for ind in range(len(matFiles)-1):
        dataIn, dataOut, Xf, Yf, Zf = prepareDataInFromCFD(ind, matFiles=matFiles, B=B, dt=data.parameters['dt'])

        dataNN[:, :, :, :, 0:9] = dataIn[:, :, :, :, 0:9]
        gen = net.predict(dataNN)
        dataNN[:, :, :, :, 9:13] = gen[:, :, :, :]

        plotResult(pathResults, gen[:,:,:,-5,:], dataIn[:,:,:,-5,:], dataOut[:,:,:,-5,:], ind)

        vtk(pathResults / Path('result_' + str(ind) + '.vtu'), B, Xf, Yf, Zf, gen[0,:,:,:,0], gen[0,:,:,:,1], gen[0,:,:,:,2], gen[0,:,:,:,3])
        vtk(pathResults / Path('result_CFD_' + str(ind) + '.vtu'), B, Xf, Yf, Zf, dataOut[0, :, :, :, 0], dataOut[0, :, :, :, 1], dataOut[0, :, :, :, 2], dataOut[0, :, :, :, 3])
        vtk(pathResults / Path('difference_' + str(ind) + '.vtu'), B, Xf, Yf, Zf,
            gen[0, :, :, :, 0] - dataOut[0, :, :, :, 0],
            gen[0, :, :, :, 1] - dataOut[0, :, :, :, 1],
            gen[0, :, :, :, 2] - dataOut[0, :, :, :, 2],
            gen[0, :, :, :, 3] - dataOut[0, :, :, :, 3]
            )

