import scipy.io
import numpy as np
import matplotlib.pyplot as plt


class meshDeformation3D:

    def __init__(self, fileName):
        mat = scipy.io.loadmat(fileName, struct_as_record=False)
        mesh = mat['mesh'][0, 0]
        self.X0 = mesh.X
        self.Y0 = mesh.Y
        self.Z0 = mesh.Z
        self.BF = mesh.blend3D
        self.BF = np.expand_dims(self.BF, axis=-1)
        self.nBody = len(self.BF[0, 0, 0, :])
        self.nx, self.ny, self.nz = np.shape(self.X0)

    def computeMesh(self, xt, yt, zt):
        xField = np.zeros(np.shape(self.X0))
        yField = np.zeros(np.shape(self.X0))
        zField = np.zeros(np.shape(self.X0))
        for body in range(self.nBody):
            xField = xField + self.BF[:, :, :, body] * xt[body]
            yField = yField + self.BF[:, :, :, body] * yt[body]
            zField = zField + self.BF[:, :, :, body] * zt[body]

        return self.X0 + xField, self.Y0 + yField, self.Z0 + zField

    def computeTiltMesh(self, xt, yt, zt):
        nx, ny, nz = np.shape(self.X0)

        xField = np.zeros(np.shape(self.X0))
        yField = np.zeros(np.shape(self.X0))

        for k in range(nz):
            for body in range(self.nBody):
                z = k / (nz - 1.0)
                xField[:, :, k] = xField[:, :, k] + self.BF[:, :, k, body] * xt[body] * z
                yField[:, :, k] = yField[:, :, k] + self.BF[:, :, k, body] * yt[body] * z

        return self.X0 + xField, self.Y0 + yField, self.Z0

    def computeB(self):
        nx, ny, nz, nbf = np.shape(self.BF)
        B = np.zeros((nx, ny, nz))
        tol = 1e-3
        for i in range(self.nBody):
            B[np.abs(self.BF[:, :, :, i] - 1) < tol] = 1

        return B

    def showBf(self, body):
        plt.figure()
        plt.pcolormesh(self.BF[:, :, 0, body])
        plt.show()


if __name__ == "__main__":
    md = meshDeformation3D('../../data/TrainingData/def3D.mat')
    # X, Y = md.computeMesh([0.0, 0, 0, 0], [0.0, 0, 0, 0])
    X, Y, Z = md.computeMesh(np.zeros(20), np.zeros(20), np.zeros(20))
    print(np.shape(X))

    md.showBf(0)

    plt.figure()
    for i in range(len(X[:, 1, 0])):
        plt.plot(X[i, :, 0], Y[i, :, 0], color='k')
    for j in range(len(X[1, :, 0])):
        plt.plot(X[:, j, 0], Y[:, j, 0], color='k')

    plt.show()
