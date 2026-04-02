import numpy as np
import keras
from matplotlib import pyplot as plt
from pathlib import Path
from meshDeformation3D import meshDeformation3D as meshDeformation
# import structure_mass_spring as structure
from toVtk import vtk, vtkBoundary


def saveData(resultPath, name, data):
    n = len(data)
    m = len(data[0])
    dataOut = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dataOut[i, j] = data[i][j]

    np.savetxt(resultPath / Path(name + '.txt'), dataOut, delimiter=',')


def plotResult(path, cascadeData, iter, ind=0):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams.update({'font.size': 15})

    X = cascadeData[0, :, :, 0]
    Y = cascadeData[0, :, :, 1]
    B = cascadeData[0, :, :, 4]
    u = cascadeData[0, :, :, 5]
    v = cascadeData[0, :, :, 6]
    p = cascadeData[0, :, :, 7]
    uv = np.sqrt(u ** 2 + v ** 2)
    uv[B == 1] = 0

    velMin = np.amin(uv)
    velMax = np.amax(uv)
    pMin = np.amin(p)
    pMax = np.amax(p)
    nLevels = 20

    fig = plt.figure(figsize=(18, 16), )
    spec = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(spec[0, 0])
    plt.title('DNN - Velocity', fontsize=20)
    ax0.axes.xaxis.set_visible(False)
    ax0.axes.yaxis.set_visible(False)

    ax1 = fig.add_subplot(spec[0, 1])
    plt.title('DNN - Pressure', fontsize=20)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)

    # ax2 = fig.add_subplot(spec[0, 2])
    # plt.title('mesh', fontsize=20)
    # ax1.axes.xaxis.set_visible(False)
    # ax1.axes.yaxis.set_visible(False)

    ax = ax0
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, uv, 20, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    ax = ax1
    ax.set_aspect('equal', adjustable='box')
    im = ax.contourf(X, Y, p, 20, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    plt.colorbar(im, cax=cax, format='%.2f')

    # ax = ax2
    # ax.set_aspect('equal', adjustable='box')
    # for i in range(len(X[:, 1])):
    #     ax.plot(X[i, :], Y[i, :], color='k', linewidth=0.3)
    # for j in range(len(X[1, :])):
    #     ax.plot(X[:, j], Y[:, j], color='k', linewidth=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(path / Path('id_' + str(iter) + '.png'), dpi=100)
    plt.close(fig)


def calculatelift(X, B, gen):
    p = gen[:, :, 2]
    nx, ny = np.shape(B)
    nyBottom = int(ny / 2) - 1
    nyTop = nyBottom + 1
    eps = 0.999
    lift = 0
    for i in range(nx-1):
        if (B[i, nyBottom] > eps and B[i, nyTop] > eps and B[i+1, nyBottom] > eps and B[i+1, nyTop] > eps):
            dx = X[i + 1, nyBottom] - X[i, nyBottom]
            pc = (p[i + 1, nyBottom] + p[i, nyBottom]) / 2
            lift += dx * pc

            dx = X[i + 1, nyTop] - X[i, nyTop]
            pc = (p[i + 1, nyTop] + p[i, nyTop]) / 2
            lift -= dx * pc

    return lift


def findIndexes(B):
    nx, ny, nz = np.shape(B)
    nxCenter = int(nx / 2)
    nzCenter = int(nz / 2)
    indexes = []
    indexes.append(0)
    for j in range(ny):
        if B[nxCenter, j, nzCenter] == 1:
            indexes.append(j)
    indexes.append(ny-1)

    return range(indexes[0], indexes[1]+1), range(indexes[2], indexes[3]+1), range(indexes[4], indexes[5]+1), range(indexes[6], indexes[7]+1)


def mixResults(topData, bottomData):
    spec, m, n, o, dim = np.shape(topData)
    E = np.zeros((spec, m, n, o, dim))
    for j in range(n):
        E[0:1, :, j, :, :] = j / (n - 1.0)
    return topData * (1 - E) + bottomData * E


def handleCascadeData(dataIn, J2bottom, Jbottom, Jtop, J2top):
    nProfiles = len(dataIn[:, 0, 0, 0, 0])
    nSpaces = nProfiles
    spacesData = []
    profileIndexes = []
    dy = (dataIn[0, 0, -1, 0, 1] - dataIn[0, 0, 0, 0, 1]) / 4
    # collection
    for i in range(nSpaces):
        indexOfTop2Profile = (i + 2) % nProfiles
        indexOfTopProfile = (i + 1) % nProfiles
        indexOfBottomProfile = i % nProfiles
        indexOfBottom2Profile = (i - 1) % nProfiles
        dataFromTopProfile = dataIn[indexOfTopProfile:indexOfTopProfile+1, :, Jbottom, :, :]
        dataFromBottomProfile = dataIn[indexOfBottomProfile:indexOfBottomProfile+1, :, Jtop, :, :]
        # mixedData = (dataFromTopProfile + dataFromBottomProfile) / 2
        mixedData = mixResults(dataFromTopProfile, dataFromBottomProfile)
        mixedData[:, :, :, :, 1] = dataFromBottomProfile[:, :, :, :, 1] + dy * i  # y coordinates
        spacesData.append(mixedData)
        profileIndexes.append([indexOfTop2Profile, indexOfTopProfile, indexOfBottomProfile, indexOfBottom2Profile])

    # redistribution
    for i in range(nProfiles):
        dataIn[i:i+1, :, J2top, :, 7:11] = spacesData[(profileIndexes[i][0] - 1) % nProfiles][..., 7:11]
        dataIn[i:i+1, :, Jtop, :, 7:11] = spacesData[(profileIndexes[i][1] - 1) % nProfiles][..., 7:11]
        dataIn[i:i+1, :, Jbottom, :, 7:11] = spacesData[(profileIndexes[i][2] - 1) % nProfiles][..., 7:11]
        dataIn[i:i+1, :, J2bottom, :, 7:11] = spacesData[(profileIndexes[i][3] - 1) % nProfiles][..., 7:11]

    return np.concatenate(spacesData, axis=2)


def getPossitions(t, i, amplitude, frequency, phaseShift):
    nProfiles = len(amplitude)
    iPlus = (i + 1) % nProfiles
    iMinus = (i - 1) % nProfiles
    return (np.array([0, 0, 0]),
            np.array(
                [amplitude[iMinus] * np.sin(frequency[iMinus] * t + phaseShift[iMinus]),
                 amplitude[i] * np.sin(frequency[i] * t + phaseShift[i]),
                 amplitude[iPlus] * np.sin(frequency[iPlus] * t + phaseShift[iPlus])]),
            np.array([0, 0, 0]))


if __name__ == "__main__":
    saveFigures = True
    dataDirs = ['../../data/TrainingData/run_3D']
    path = Path('../../data/net22_3D')
    pathResults = path / Path('results_CNN')
    pathResults.mkdir(exist_ok=True)
    md = meshDeformation(Path(dataDirs[0]).parents[0] / Path('def3D.mat'))
    B = md.computeB()
    J0, J1, J2, J3 = findIndexes(B)
    nPart = len(J1)

    net = keras.models.load_model(path / Path("model.keras"))

    nProfiles = 7
    T = 10
    dt = 0.25
    Uo = 1
    dtStructure = 0.02
    mass = 30.0
    f = 0.1
    damping = 0

    amplitude = [0, 0.15, 0.07, 0.15, 0.09, 0.11, 0.1]
    frequency = [0, 0.55, 0.65, 0.6, 0.45, 0.35, 0.2]
    phaseShift = [0, 3.14, 3.14 / 4, 0, 3.14 / 2, 0, 3.14 / 4]

    # structs = []
    # for profile in range(nProfiles):
    #     struct = structure.structureSolver(mass, f, damping)
    #     struct.setDt(dt, dtStructure)
    #     struct.setInitCondition(0, 0.01*np.random.uniform(-1, 1))
    #     structs.append(struct)

    nx, ny, nz = np.shape(md.X0)
    dataIn = np.zeros([nProfiles, nx, ny, nz, 11], dtype=np.float32)

    for i in range(nProfiles):
        xtn, ytn, ztn = getPossitions(0, i, amplitude, frequency, phaseShift)
        Xn, Yn, Zn = md.computeTiltMesh(xtn, ytn, ztn)
        dataIn[i, :, :, :, 0] = Xn
        dataIn[i, :, :, :, 1] = Yn
        dataIn[i, :, :, :, 2] = Zn
        dataIn[i, :, :, :, 6] = B
        dataIn[i, :, :, :, 7] = Uo

    y_in_time = []
    lifts_in_time = []
    for iter, t in enumerate(np.arange(dt, T, dt)):
        print("Time: " + str(round(t, 3)) + ", Iter: " + str(iter))

        # compute deformed mesh
        for i in range(nProfiles):
            xtn, ytn, ztn = getPossitions(t, i, amplitude, frequency, phaseShift)
            Xn1, Yn1, Zn1 = md.computeTiltMesh(xtn, ytn, ztn)
            uMesh = (Xn1 - dataIn[i, :, :, :, 0]) / dt
            vMesh = (Yn1 - dataIn[i, :, :, :, 1]) / dt
            wMesh = (Zn1 - dataIn[i, :, :, :, 2]) / dt

            dataIn[i, :, :, :, 3] = uMesh
            dataIn[i, :, :, :, 4] = vMesh
            dataIn[i, :, :, :, 5] = wMesh

        # predict solution
        gen = net.predict(dataIn)

        for i in range(nProfiles):
            dataIn[i, :, :, :, 0] += dataIn[i, :, :, :, 3] * dt
            dataIn[i, :, :, :, 1] += dataIn[i, :, :, :, 4] * dt
            dataIn[i, :, :, :, 2] += dataIn[i, :, :, :, 5] * dt
            dataIn[i, :, :, :, 7:11] = gen[i, :, :, :, :]

        # build cascade
        cascadeData = handleCascadeData(dataIn, J0, J1, J2, J3)

        # # calculate lift and update structure
        # y = []
        # lifts = []
        # for i in range(nProfiles):
        #     lift = calculatelift(dataIn[i, :, :, 0], dataIn[i, :, :, 4], dataIn[i, :, :, 5:8])
        #     structs[i].step(lift)
        #     # store
        #     lifts.append(lift)
        #     y.append(structs[i].get_y())
        # y_in_time.append(y)
        # lifts_in_time.append(lifts)

        if iter % 2 == 0 and saveFigures == True:
            vtkBoundary(pathResults / Path('boundary_' + str(iter) + '.vtu'),
                        cascadeData[0, :, :, :, 6],
                        cascadeData[0, :, :, :, 0],
                        cascadeData[0, :, :, :, 1],
                        cascadeData[0, :, :, :, 2],
                        cascadeData[0, :, :, :, 10])
            vtk(pathResults / Path('result_' + str(iter) + '.vtu'),
                cascadeData[0, :, :, :, 6],
                cascadeData[0, :, :, :, 0],
                cascadeData[0, :, :, :, 1],
                cascadeData[0, :, :, :, 2],
                cascadeData[0, :, :, :, 7],
                cascadeData[0, :, :, :, 8],
                cascadeData[0, :, :, :, 9],
                cascadeData[0, :, :, :, 10])

    # saveData(pathResults, 'position', y_in_time)
    # saveData(pathResults, 'lift', lifts_in_time)

