import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter


def smooth_flow_field(U, V, P, sigma_UV=2.0, sigma_P=2.0, preserve_div=True):
    """
    Vyhladí rychlostní pole (U,V) a tlakové pole P.

    Parametry:
        U, V : 2D pole rychlostí
        P : 2D pole tlaku
        sigma_UV : sigma pro vyhlazení U a V
        sigma_P : sigma pro vyhlazení P
        preserve_div : zda zachovat divergenci rychlostí

    Výstup:
        U_s, V_s, P_s : vyhlazené pole
    """

    # --- vyhlazení rychlostí ---
    U_s = gaussian_filter(U, sigma=sigma_UV)
    V_s = gaussian_filter(V, sigma=sigma_UV)

    if preserve_div:
        # zachování divergence (doporučeno pro incompresible flow)
        dUdx = np.gradient(U, axis=1)
        dVdy = np.gradient(V, axis=0)
        div_orig = dUdx + dVdy

        dUdx_s = np.gradient(U_s, axis=1)
        dVdy_s = np.gradient(V_s, axis=0)
        div_s = dUdx_s + dVdy_s

        correction = div_s - div_orig
        U_s -= correction / 2.0
        V_s -= correction / 2.0

    # --- vyhlazení tlaku ---
    P_s = gaussian_filter(P, sigma=sigma_P)

    return U_s, V_s, P_s

def compute_streamlines_general(X, Y, U, V, x0, y0, max_time=10, max_step=0.01):

    # Převedeme síť na seznam bodů
    pts = np.column_stack((X.flatten(), Y.flatten()))

    # Interpolátory pro U(x,y), V(x,y)
    u_interp = LinearNDInterpolator(pts, U.flatten())
    v_interp = LinearNDInterpolator(pts, V.flatten())

    # RHS ODE: d[x,y]/ds = [u(x,y), v(x,y)]
    def rhs(s, xy):
        x, y = xy
        u = u_interp(x, y)
        v = v_interp(x, y)

        # mimo doménu → zastavit
        if np.isnan(u) or np.isnan(v):
            return [0, 0]
        return [u, v]

    streamlines = []
    for xs, ys in zip(x0, y0):
        sol = solve_ivp(rhs, [0, max_time], [xs, ys], max_step=max_step)
        streamlines.append(sol.y)

    return streamlines

def plotResult(axs, gen, dataIn):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    # smažeme starý obsah
    for ax in axs:
        ax.clear()

    # první řádek: rychlost
    axs[0].set_title("Velocity [1]", fontsize=14)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].contourf(X, Y + dy, uv, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    axs[0].contourf(X, Y, uv, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    axs[0].contourf(X, Y-dy, uv, cmap='jet', levels=np.linspace(velMin, velMax, nLevels))
    axs[0].quiver(X[::4,::4],Y[::4,::4],u[::4,::4],v[::4,::4], scale=50)
    axs[0].axis("off")
    #
    # x0 = X[1, ::3]
    # y0 = Y[1, ::3]
    # streamlines = compute_streamlines_general(X, Y, u, v, x0, y0)
    # for s in streamlines:
    #     axs[0].plot(s[0], s[1], color="black")

    # druhý řádek: tlak
    axs[1].set_title("Pressure [1]", fontsize=14)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].contourf(X, Y + dy, p, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    axs[1].contourf(X, Y, p, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    axs[1].contourf(X, Y - dy, p, cmap='jet', levels=np.linspace(pMin, pMax, nLevels))
    axs[1].axis("off")


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

    scales = np.load(os.path.join(path, "scales.npy"), allow_pickle=True).item()
    in_min = scales["in_min"]
    in_max = scales["in_max"]
    out_min = scales["out_min"]
    out_max = scales["out_max"]
    dataOut = dataOut * (out_max - out_min) + out_min

    i = 64

    # figure se 2 řádky a 3 sloupci
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs = axs.flatten()

    # inicializace
    gen = net.predict(dataIn[i:i+1, :, :, :])
    gen = gen * (out_max - out_min) + out_min
    plotResult(axs, gen[0, :, :, :], dataIn[i, :, :, :])

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
        gen = gen * (out_max - out_min) + out_min

        plotResult(axs, gen[0, :, :, :], dataIn[i, :, :, :])
        fig.canvas.draw_idle()

    slider_re.on_changed(update)
    slider_alfa.on_changed(update)

    plt.show()
