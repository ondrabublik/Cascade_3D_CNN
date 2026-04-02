import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt

def plotResult(gen, dataIn, alfa, Re):

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs = axs.flatten()

    X = dataIn[:, :, 0]
    Y = dataIn[:, :, 1]
    dy = Y[0,-1] - Y[0,0]

    u = gen[:, :, 0]
    v = gen[:, :, 1]
    p = gen[:, :, 2]
    uv = np.sqrt(u ** 2 + v ** 2)

    velMin = 0
    velMax = np.amax([np.amax(uv)])
    pMin = np.amin([np.amin(p)])
    pMax = np.amax([np.amax(p)])
    nLevels = 20

    for ax in axs:
        ax.clear()

    axs[0].set_title("Velocity [1], alpha =  " + "{}°".format(round(alfa)) + ", Re =  " + "{}".format(round(Re)), fontsize=14)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].contourf(X, Y + dy, uv, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    axs[0].contourf(X, Y, uv, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    axs[0].contourf(X, Y-dy, uv, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    axs[0].quiver(X[::4,::4],Y[::4,::4],u[::4,::4],v[::4,::4], scale=50)
    axs[0].axis("off")

    axs[1].set_title("Pressure [1]", fontsize=14)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].contourf(X, Y + dy, p, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    axs[1].contourf(X, Y, p, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    axs[1].contourf(X, Y - dy, p, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    axs[1].axis("off")

    return fig



if __name__ == "__main__":
    dataDirs = ['../../data/training_data/test_2D_v2']
    path = Path('../../data/training_data/test_2D_v2')
    pathResultsAlfa = path / Path('results_walk_alfa')
    pathResultsAlfa.mkdir(exist_ok=True)
    pathResultsRe = path / Path('results_walk_Re')
    pathResultsRe.mkdir(exist_ok=True)

    from UNetDev2D_periodic import AddBC

    net = keras.models.load_model(path / Path("model.keras"), safe_mode=False, custom_objects={
        'slice': slice,
        'tf': tf,
        'AddBC': AddBC})

    dataIn = np.load(os.path.join(path, "dataIn.npy"))
    dataOut = np.load(os.path.join(path, "dataOut.npy"))
    nSpec, nx, ny, dimIn = np.shape(dataIn)

    scales = np.load(os.path.join(path, "scales.npy"), allow_pickle=True).item()
    in_min = scales["in_min"]
    in_max = scales["in_max"]
    out_min = scales["out_min"]
    out_max = scales["out_max"]
    dataOut = dataOut * (out_max - out_min) + out_min

    i = 64

    # inicializace
    gen = net.predict(dataIn[i:i + 1, :, :, :])
    gen = gen * (out_max - out_min) + out_min

    Re = 0.3
    alfa = np.linspace(1,0,30)

    for j in range(len(alfa)):
        dataIn[i:i + 1, :, :, 3] = Re
        dataIn[i:i + 1, :, :, 4] = alfa[j]
        gen = net.predict(dataIn[i:i + 1, :, :, :])
        gen = gen * (out_max - out_min) + out_min

        fig = plotResult(gen[0, :, :, :], dataIn[i, :, :, :], alfa[j] * (in_max[4]-in_min[4]) + in_min[4], Re * (in_max[3]-in_min[3]) + in_min[3])
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(pathResultsAlfa / Path('id_alfa' + str(j) + '.png'), dpi=150)
        plt.close(fig)

    Re = np.linspace(0,1,30)
    alfa = 0.7

    for j in range(len(Re)):
        dataIn[i:i + 1, :, :, 3] = Re[j]
        dataIn[i:i + 1, :, :, 4] = alfa
        gen = net.predict(dataIn[i:i + 1, :, :, :])
        gen = gen * (out_max - out_min) + out_min

        fig = plotResult(gen[0, :, :, :], dataIn[i, :, :, :], alfa * (in_max[4]-in_min[4]) + in_min[4], Re[j] * (in_max[3]-in_min[3]) + in_min[3])
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(pathResultsRe / Path('id_Re' + str(j) + '.png'), dpi=150)
        plt.close(fig)

