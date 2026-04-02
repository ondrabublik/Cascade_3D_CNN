import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from pathlib import Path
import os


def plotResult(path, gen, dataIn, dataOut, ind):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams.update({'font.size': 15})

    X = dataIn[:, :, 0]
    Y = dataIn[:, :, 1]
    dy = Y[0,-1] - Y[0,0]

    u = gen[:, :, 0]
    v = gen[:, :, 1]
    p = gen[:, :, 2]
    uv = np.sqrt(u ** 2 + v ** 2)

    u0 = dataOut[:, :, 0]
    v0 = dataOut[:, :, 1]
    p0 = dataOut[:, :, 2]
    uv0 = np.sqrt(u0 ** 2 + v0 ** 2)

    velMin = np.amin([np.amin(uv), np.amin(uv0)])
    velMax = np.amax([np.amax(uv), np.amax(uv0)])
    pMin = np.amin([np.amin(p), np.amin(p0)])
    pMax = np.amax([np.amax(p), np.amax(p0)])
    nLevels = 20

    fig = plt.figure(figsize=(18, 16), )
    spec = fig.add_gridspec(2, 3)
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

    ax = ax0
    ax.set_aspect('equal', adjustable='box')
    ax.contourf(X, Y, uv, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    ax.contourf(X, Y-dy, uv, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))

    ax = ax01
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, uv0, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    im = ax.contourf(X, Y-dy, uv0, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    ax = ax02
    errVel = abs(uv - uv0)
    maxErrVel = np.amax(errVel)
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, errVel, 20, cmap='jet', levels=np.linspace(0, maxErrVel, nLevels))
    im = ax.contourf(X, Y-dy, errVel, 20, cmap='jet', levels=np.linspace(0, maxErrVel, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    ax = ax10
    ax.set_aspect('equal', adjustable='box')
    ax.contourf(X, Y, p, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    ax.contourf(X, Y-dy, p, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))

    ax = ax11
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, p0, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    im = ax.contourf(X, Y-dy, p0, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    ax = ax12
    errP = abs(p - p0)  # / p0 * 100
    maxErrP = np.amax(errP)
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, errP, 20, cmap='jet', levels=np.linspace(0, maxErrP, nLevels))
    im = ax.contourf(X, Y-dy, errP, 20, cmap='jet', levels=np.linspace(0, maxErrP, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')


    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(path / Path('id_' + str(ind) + '.png'), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    dataDirs = ['../../data/training_data/test_2D_v2']
    path = Path('../../data/training_data/test_2D_v2')
    pathResults = path / Path('results')
    pathResults.mkdir(exist_ok=True)

    from UNetDev2D_periodic import AddBC

    net = keras.models.load_model(path / Path("model.keras"), safe_mode=False, custom_objects={
        'slice': slice,
        'tf': tf,
        'AddBC': AddBC})

    dataIn = np.load(os.path.join(path, "dataIn.npy"))
    dataOut = np.load(os.path.join(path, "dataOut.npy"))
    nSpec, nx, ny, dimIn = np.shape(dataIn)

    for i in range(nSpec):
        gen = net.predict(dataIn[i:i+1,:,:,:])

        plotResult(pathResults, gen[0,:,:,:], dataIn[i,:,:,:], dataOut[i,:,:,:], i)

