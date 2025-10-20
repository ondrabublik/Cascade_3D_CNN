import numpy as np
from pathlib import Path
import scipy.io
import re


class DataLoader:

	def __init__(self, dir):
		self.dataDir = dir
		self.readData()

	def getData(self, i, dt):
		mat = scipy.io.loadmat(self.sorted_mat_files[i])['data']
		nextMat = scipy.io.loadmat(self.sorted_mat_files[i + 1])['data']
		nx, ny = np.shape(mat['X'][0][0])

		data = np.zeros((nx, ny, 11))

		data[0:nx, 0:ny, 0] = mat['X'][0][0]
		data[0:nx, 0:ny, 1] = mat['Y'][0][0]
		data[0:nx, 0:ny, 2] = (nextMat['X'][0][0] - mat['X'][0][0]) / dt
		data[0:nx, 0:ny, 3] = (nextMat['Y'][0][0] - mat['Y'][0][0]) / dt
		data[0:nx, 0:ny, 5] = mat['u'][0][0]
		data[0:nx, 0:ny, 6] = mat['v'][0][0]
		data[0:nx, 0:ny, 7] = mat['p'][0][0]

		data[0:nx, 0:ny, 8] = nextMat['u'][0][0]
		data[0:nx, 0:ny, 9] = nextMat['v'][0][0]
		data[0:nx, 0:ny, 10] = nextMat['p'][0][0]

		return data

	def readData(self):
		mat_files = [f for f in Path(self.dataDir).iterdir()]
		self.sorted_mat_files = sorted(mat_files, key=lambda filename: int(re.search(r'\d+', filename.name).group()))

		mat = scipy.io.loadmat(self.sorted_mat_files[0])['data']
		nx, ny = np.shape(mat['X'][0][0])
		self.nx = nx
		self.ny = ny
		self.nSamples = len(mat_files) - 1

