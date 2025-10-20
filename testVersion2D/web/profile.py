import numpy as np
import matplotlib.pyplot as plt
from math import comb

def generateProfileBezier3(a, b, c, d, n):
    a0 = (1 - 0.2) / 2
    c0 = 0.5 / 2

    p1 = np.array([
        [0, 0.5],
        [0, 0.55],
        [0.2, a + a0],
        [0.5, c + c0],
        [1, 0.05],
        [1, 0]
    ])

    p2 = np.array([
        [0, 0.5],
        [0, 0.45],
        [0.2, b + a0],
        [0.5, d + c0],
        [1, -0.05],
        [1, 0]
    ])

    xyd = bezier_curve(n, p1)
    xyh = bezier_curve(n, p2)

    # Vykreslení výsledků
    plt.figure()
    plt.plot(p1[:, 0], p1[:, 1], 'or')
    plt.plot(p2[:, 0], p2[:, 1], 'ob')
    plt.plot(xyd[:, 0], xyd[:, 1], 'r')
    plt.plot(xyh[:, 0], xyh[:, 1], 'b')
    plt.axis('equal')
    plt.show()

    return xyd, xyh


def bezier_curve(N, P):
    """
    Vytvoří Bezierovu křivku z daných řídicích bodů P.
    N je počet bodů, které se mají spočítat.
    """
    Np = P.shape[0]
    u = np.linspace(0, 1, N)
    B = np.zeros((N, Np))

    for i in range(Np):
        B[:, i] = comb(Np - 1, i) * (u ** i) * ((1 - u) ** (Np - 1 - i))

    S = B @ P
    return S[:, [0, 1]]


# --- Příklad použití ---
if __name__ == "__main__":
    # parametry pro profil
    xyd, xyh = generateProfileBezier3(a=0.5, b=0.1, c=0.15, d=0.05, n=34)
