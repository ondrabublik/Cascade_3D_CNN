import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
import os


def plotResult(axs, gen, dataIn, dataOut):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    # smažeme starý obsah
    for ax in axs:
        ax.clear()

    # první řádek: rychlost
    axs[0].set_title("DNN - Velocity", fontsize=14)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].contourf(X, Y, uv, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    axs[0].contourf(X, Y-dy, uv, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    axs[0].axis("off")

    axs[1].set_title("CFD - Velocity", fontsize=14)
    axs[1].set_aspect('equal', adjustable='box')
    im = axs[1].contourf(X, Y, uv0, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    im = axs[1].contourf(X, Y-dy, uv0, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    axs[1].axis("off")
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    axs[2].set_title("Absolute error - Velocity", fontsize=14)
    errVel = abs(uv - uv0)
    maxErrVel = np.amax(errVel)
    axs[2].set_aspect('equal', adjustable='box')
    im = axs[2].contourf(X, Y, errVel, 20, cmap='jet', levels=np.linspace(0, maxErrVel, nLevels))
    im = axs[2].contourf(X, Y-dy, errVel, 20, cmap='jet', levels=np.linspace(0, maxErrVel, nLevels))
    axs[2].axis("off")
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    # druhý řádek: tlak
    axs[3].set_title("DNN - Pressure", fontsize=14)
    axs[3].set_aspect('equal', adjustable='box')
    axs[3].contourf(X, Y, p, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    axs[3].contourf(X, Y-dy, p, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    axs[3].axis("off")

    axs[4].set_title("CFD - Pressure", fontsize=14)
    axs[4].set_aspect('equal', adjustable='box')
    im = axs[4].contourf(X, Y, p0, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    im = axs[4].contourf(X, Y-dy, p0, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    axs[4].axis("off")
    divider = make_axes_locatable(axs[4])
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    axs[5].set_title("Absolute error - Pressure", fontsize=14)
    errP = abs(p - p0)
    maxErrP = np.amax(errP)
    axs[5].set_aspect('equal', adjustable='box')
    im = axs[5].contourf(X, Y, errP, 20, cmap='jet', levels=np.linspace(0, maxErrP, nLevels))
    im = axs[5].contourf(X, Y-dy, errP, 20, cmap='jet', levels=np.linspace(0, maxErrP, nLevels))
    axs[5].axis("off")
    divider = make_axes_locatable(axs[5])
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')


if __name__ == "__main__":
    dataDirs = ['../../data/training_data/test_2D']
    path = Path('../../data/training_data/test_2D')
    pathResults = path / Path('results')
    pathResults.mkdir(exist_ok=True)

    net = keras.models.load_model(path / Path("model.keras"), safe_mode=False, custom_objects={
        'slice': slice,
        'tf': tf})

    dataIn = np.load(os.path.join(path, "dataIn.npy"))
    dataOut = np.load(os.path.join(path, "dataOut.npy"))
    nSpec, nx, ny, dimIn = np.shape(dataIn)

    i = 1

    # figure se 2 řádky a 3 sloupci
    fig, axs = plt.subplots(2, 3, figsize=(14, 10))
    axs = axs.flatten()

    # inicializace
    gen = net.predict(dataIn[i:i+1, :, :, :])
    plotResult(axs, gen[0, :, :, :], dataIn[i, :, :, :], dataOut[i, :, :, :])

    # posuvníky pod grafem
    ax_re = plt.axes([0.2, 0.05, 0.65, 0.03])
    ax_alfa = plt.axes([0.2, 0.01, 0.65, 0.03])

    slider_re = Slider(ax_re, 'Re', 0, 1.0, valinit=dataIn[i, 0, 0, 3])
    slider_alfa = Slider(ax_alfa, 'Alfa', 0, 1.0, valinit=dataIn[i, 0, 0, 4])

    def update(val):
        # upravíme dataIn podle sliderů
        dataIn[i:i+1, :, :, 3] = slider_re.val
        dataIn[i:i+1, :, :, 4] = slider_alfa.val
        gen = net.predict(dataIn[i:i+1, :, :, :])
        plotResult(axs, gen[0, :, :, :], dataIn[i, :, :, :], dataOut[i, :, :, :])
        fig.canvas.draw_idle()

    slider_re.on_changed(update)
    slider_alfa.on_changed(update)

    plt.show()
