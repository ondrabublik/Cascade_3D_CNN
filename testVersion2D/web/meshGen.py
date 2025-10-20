import numpy as np

def mesh_gen(xyd, xyh):
    H = 1.0
    alfa = 0.0

    m = 64
    n = 90

    # spojení horní a dolní části profilu
    prof = np.vstack((xyh[::-1, :], xyd[1:, :]))

    # transformace profilu
    alfa = alfa / 180.0 * np.pi
    profN = np.zeros_like(prof)
    profN[:, 0] = np.cos(alfa) * prof[:, 0] - np.sin(alfa) * prof[:, 1]
    profN[:, 1] = np.sin(alfa) * prof[:, 0] + np.cos(alfa) * prof[:, 1]
    prof = profN

    # profil musí mít lichý počet bodů
    n_prof = prof.shape[0]
    npr = n
    Id = np.arange(npr - 1, n_prof)
    Ih = np.arange(npr - 1, -1, -1)

    # horní a dolní část profilu
    Xh = prof[Ih, :]
    Xd = prof[Id, :]

    # přední napojení
    np_points = 48
    t = np.linspace(0, 1, np_points)
    P0 = Xh[0, :]
    P2 = P0 - np.array([H / 2, 0])
    Xp = np.zeros((np_points, 2))
    Xp[:, 0] = (1 - t) * P2[0] + t * P0[0]
    Xp[:, 1] = (1 - t) * P2[1] + t * P0[1]

    # zadní napojení
    nz = 56  # fix(n/4)+2
    t = np.linspace(0, 2, nz)
    L = -0.5
    P0 = Xh[npr - 1, :]
    P2 = P0 + np.array([H - 0.5, L])
    Xz = np.zeros((nz, 2))
    Xz[:, 0] = (1 - t) * P0[0] + t * P2[0]
    Xz[:, 1] = (1 - t) * P0[1] + t * P2[1]

    # tvorba sítě
    nx = nz + npr + np_points - 2
    ny = m

    X = np.zeros((nx, ny))
    Y = np.zeros((nx, ny))
    B = np.zeros((nx, ny))

    # přední část
    for i in range(np_points - 1):
        X[i, 0] = Xp[i, 0]
        X[i, ny - 1] = Xp[i, 0]
        Y[i, 0] = Xp[i, 1]
        Y[i, ny - 1] = Xp[i, 1] + H

    # profil
    for i in range(npr):
        X[i + np_points - 1, 0] = Xd[i, 0]
        X[i + np_points - 1, ny - 1] = Xh[i, 0]
        Y[i + np_points - 1, 0] = Xd[i, 1]
        Y[i + np_points - 1, ny - 1] = Xh[i, 1] + H
        B[i + np_points - 1, 0] = 1
        B[i + np_points - 1, ny - 1] = 1

    # zadní část
    for i in range(1, nz):
        X[i + np_points + npr - 2, 0] = Xz[i, 0]
        X[i + np_points + npr - 2, ny - 1] = Xz[i, 0]
        Y[i + np_points + npr - 2, 0] = Xz[i, 1]
        Y[i + np_points + npr - 2, ny - 1] = Xz[i, 1] + H

    # vstup a výstup
    for j in range(1, ny - 1):
        t = j / (ny - 1)
        X[0, j] = (1 - t) * X[0, 0] + t * X[0, ny - 1]
        Y[0, j] = (1 - t) * Y[0, 0] + t * Y[0, ny - 1]

        X[nx - 1, j] = (1 - t) * X[nx - 1, 0] + t * X[nx - 1, ny - 1]
        Y[nx - 1, j] = (1 - t) * Y[nx - 1, 0] + t * Y[nx - 1, ny - 1]

    # Volání generátorů sítě
    X, Y = AlgebGen(X, Y)
    X, Y = ElipticGen4(X, Y)

    return X, Y, B


# --- Předpokládané pomocné funkce ---
def AlgebGen(X, Y):
    n1 = X.shape[0]
    n2 = X.shape[1]

    # Normalizované parametry F a N
    F = np.linspace(0, 1, n1)
    N = np.linspace(0, 1, n2)

    Xp = np.zeros_like(X)
    Yp = np.zeros_like(Y)

    for i in range(n1):
        for j in range(n2):
            Xp[i, j] = ((1 - F[i]) * X[0, j] + F[i] * X[n1 - 1, j] +
                        (1 - N[j]) * X[i, 0] + N[j] * X[i, n2 - 1] -
                        (1 - F[i]) * (1 - N[j]) * X[0, 0] -
                        F[i] * (1 - N[j]) * X[n1 - 1, 0] -
                        (1 - F[i]) * N[j] * X[0, n2 - 1] -
                        F[i] * N[j] * X[n1 - 1, n2 - 1])

            Yp[i, j] = ((1 - F[i]) * Y[0, j] + F[i] * Y[n1 - 1, j] +
                        (1 - N[j]) * Y[i, 0] + N[j] * Y[i, n2 - 1] -
                        (1 - F[i]) * (1 - N[j]) * Y[0, 0] -
                        F[i] * (1 - N[j]) * Y[n1 - 1, 0] -
                        (1 - F[i]) * N[j] * Y[0, n2 - 1] -
                        F[i] * N[j] * Y[n1 - 1, n2 - 1])

    return Xp, Yp


def ElipticGen4(X, Y):
    a = 1e5
    c = 0.3

    n1 = X.shape[0]
    n2 = X.shape[1]

    xeta = np.zeros((n1, n2))
    yeta = np.zeros((n1, n2))
    xxi = np.zeros((n1, n2))
    yxi = np.zeros((n1, n2))
    Jac = np.zeros((n1, n2))
    g11 = np.zeros((n1, n2))
    g22 = np.zeros((n1, n2))
    g12 = np.zeros((n1, n2))
    xtemp = np.zeros((n1, n2))
    ytemp = np.zeros((n1, n2))

    PP = np.zeros((n1, n2))
    QQ = np.zeros((n1, n2))

    # počáteční algebrická generace
    X, Y = AlgebGen(X, Y)

    I = np.arange(1, n1 - 1)   # odpovídá 2:n1-1 v MATLABu
    J = np.arange(1, n2 - 1)   # odpovídá 2:n2-1 v MATLABu

    n = 500  # počet iterací

    for _ in range(n):
        # derivace podle eta a xi
        xeta[I[:, None], J] = (X[I[:, None], J + 1] - X[I[:, None], J - 1]) / 2
        yeta[I[:, None], J] = (Y[I[:, None], J + 1] - Y[I[:, None], J - 1]) / 2
        xxi[I[:, None], J] = (X[I + 1, J] - X[I - 1, J]) / 2
        yxi[I[:, None], J] = (Y[I + 1, J] - Y[I - 1, J]) / 2
        Jac[I[:, None], J] = xxi[I[:, None], J] * yeta[I[:, None], J] - xeta[I[:, None], J] * yxi[I[:, None], J]

        # zdrojové členy QQ
        for j in range(1, n2 - 1):
            QQ[:, j] = -a * np.sign(j - 1) * np.exp(-c * abs(j - 1))
            QQ[:, j] -= a * np.sign(j - (n2 - 1)) * np.exp(-c * abs(j - (n2 - 1)))

        # metrické koeficienty
        g11[I[:, None], J] = ((X[I + 1, J] - X[I - 1, J]) ** 2 + (Y[I + 1, J] - Y[I - 1, J]) ** 2) / 4
        g22[I[:, None], J] = ((X[I[:, None], J + 1] - X[I[:, None], J - 1]) ** 2 +
                              (Y[I[:, None], J + 1] - Y[I[:, None], J - 1]) ** 2) / 4
        g12[I[:, None], J] = ((X[I + 1, J] - X[I - 1, J]) * (X[I[:, None], J + 1] - X[I[:, None], J - 1]) +
                              (Y[I + 1, J] - Y[I - 1, J]) * (Y[I[:, None], J + 1] - Y[I[:, None], J - 1])) / 4

        # aktualizace uzlů vnitřní oblasti
        denom = 2 * (g11[I[:, None], J] + g22[I[:, None], J])

        xtemp[I[:, None], J] = (1.0 / denom) * (
            g22[I[:, None], J] * X[I + 1, J]
            - 0.5 * g12[I[:, None], J] * X[I + 1, J + 1]
            + 0.5 * g12[I[:, None], J] * X[I + 1, J - 1]
            + g11[I[:, None], J] * X[I[:, None], J + 1]
            + g11[I[:, None], J] * X[I[:, None], J - 1]
            + g22[I[:, None], J] * X[I - 1, J]
            - 0.5 * g12[I[:, None], J] * X[I - 1, J - 1]
            + 0.5 * g12[I[:, None], J] * X[I - 1, J + 1]
        ) + (Jac[I[:, None], J] ** 2) * (xxi[I[:, None], J] * PP[I[:, None], J] + xeta[I[:, None], J] * QQ[I[:, None], J])

        ytemp[I[:, None], J] = (1.0 / denom) * (
            g22[I[:, None], J] * Y[I + 1, J]
            - 0.5 * g12[I[:, None], J] * Y[I + 1, J + 1]
            + 0.5 * g12[I[:, None], J] * Y[I + 1, J - 1]
            + g11[I[:, None], J] * Y[I[:, None], J + 1]
            + g11[I[:, None], J] * Y[I[:, None], J - 1]
            + g22[I[:, None], J] * Y[I - 1, J]
            - 0.5 * g12[I[:, None], J] * Y[I - 1, J - 1]
            + 0.5 * g12[I[:, None], J] * Y[I - 1, J + 1]
        ) + (Jac[I[:, None], J] ** 2) * (yxi[I[:, None], J] * PP[I[:, None], J] + yeta[I[:, None], J] * QQ[I[:, None], J])

        # chyba (volitelně pro ladění)
        # err = np.sum((X[I[:, None], J] - xtemp[I[:, None], J]) ** 2 + (Y[I[:, None], J] - ytemp[I[:, None], J]) ** 2)

        # aktualizace uzlů
        X[I[:, None], J] = xtemp[I[:, None], J]
        Y[I[:, None], J] = ytemp[I[:, None], J]

    return X, Y

