import numpy as np
from pathlib import Path
import random
import scipy.io
import sys
import re
import json
from meshDeformation3D import meshDeformation3D


class Data:
	def __init__(self, dirs):
		self.dataDirs = dirs
		self.parentDir = Path(self.dataDirs[0]).parents[0]
		self.dataPath = self.parentDir / Path('data_3D')
		self.nSamplesTot = 0
		self.nx = 96 # TODO make programmatically
		self.ny = 48
		self.nz = 48
		self.parameters = {}
		self.scales = {}
		self.batchSize = 1
		self.nBatches = 15
		self.dimIn = 13
		self.dimOut = 4

		self.fileParameters = self.dataPath / Path('parameters.json')
		self.fileScales = self.dataPath / Path('scales.json')

		for dir in self.dataDirs:
			if not Path(dir).exists():
				sys.exit("Error: Data directory doesn't exists " + dir)

		if not self.dataPath.is_dir():
			self.dataPath.mkdir(parents=True, exist_ok=True)
			self.setParameters()
			with open(self.fileParameters, "w") as file:
				json.dump(self.parameters, file)

			self.prepare_training_data()

		try:
			f = open(self.fileScales)
			self.scales = json.load(f)
		except:
			print("Error: file not found!")
			print("file: ", self.fileScales)
			exit()

		try:
			f = open(self.fileParameters)
			self.parameters = json.load(f)
		except:
			print("Error: file not found!")
			print("file: ", self.fileParameters)
			exit()

		self.info()

	def info(self):
		print("\n-------------------------------------")
		# print('Number of the train samples: ', self.In.shape[0], "\n")
		print("-------------------------------------\n")

	def loadDataIn(self, idx):
		fileIn = self.dataPath / Path('dataIn_' + str(idx) + '.npy')
		print('Data loaded: ' + str(idx))
		return np.load(fileIn)

	def loadDataOut(self, idx):
		fileOut = self.dataPath / Path('dataOut_' + str(idx) + '.npy')
		return np.load(fileOut)

	def setParameters(self):
		self.parameters = {'Re': 5000, 'dt': 0.1}

	def setScales(self):
		Umin = Vmin = Wmin = Pmin = dXmin = dYmin = dZmin = 1e10
		Umax = Vmax = Wmax = Pmax = dXmax = dYmax = dZmax = -1e10

		for dir in self.dataDirs:
			print(dir)
			mat_files = [f for f in Path(dir).iterdir()]
			sorted_mat_files = sorted(mat_files, key=lambda filename: int(re.search(r'\d+', filename.name).group()))

			for i in range(len(sorted_mat_files)-1):
				mat = scipy.io.loadmat(sorted_mat_files[i])['data']
				nextMat = scipy.io.loadmat(sorted_mat_files[i+1])['data']
				dX = nextMat['X'][0][0] - mat['X'][0][0]
				dY = nextMat['Y'][0][0] - mat['Y'][0][0]
				dZ = nextMat['Z'][0][0] - mat['Z'][0][0]
				Umin = min(Umin, np.min(mat['U'][0][0]))
				Umax = max(Umax, np.max(mat['U'][0][0]))
				Vmin = min(Vmin, np.min(mat['V'][0][0]))
				Vmax = max(Vmax, np.max(mat['V'][0][0]))
				Wmin = min(Wmin, np.min(mat['W'][0][0]))
				Wmax = max(Wmax, np.max(mat['W'][0][0]))
				Pmin = min(Pmin, np.min(mat['P'][0][0]))
				Pmax = max(Pmax, np.max(mat['P'][0][0]))
				dXmin = min(dXmin, np.min(dX))
				dXmax = max(dXmax, np.max(dX))
				dYmin = min(dYmin, np.min(dY))
				dYmax = max(dYmax, np.max(dY))
				dZmin = min(dZmin, np.min(dZ))
				dZmax = max(dZmax, np.max(dZ))

		velMin = min(Umin, Vmin, Wmin)
		velMax = max(Umax, Vmax, Wmax)
		velMeshMin = min(dXmin/self.parameters['dt'], dYmin/self.parameters['dt'], dZmin/self.parameters['dt'])
		velMeshMax = max(dXmax/self.parameters['dt'], dYmax/self.parameters['dt'], dZmax/self.parameters['dt'])

		print('U (min/max) = ' + str(round(Umin, 2)) + ' / ' + str(round(Umax, 2)))
		print('V (min/max) = ' + str(round(Vmin, 2)) + ' / ' + str(round(Vmax, 2)))
		print('W (min/max) = ' + str(round(Wmin, 2)) + ' / ' + str(round(Wmax, 2)))
		print('P (min/max) = ' + str(round(Pmin, 2)) + ' / ' + str(round(Pmax, 2)))
		print('uMesh (min/max) = ' + str(round(dXmin/self.parameters['dt'], 8)) + ' / ' + str(round(dXmax/self.parameters['dt'], 8)))
		print('vMesh (min/max) = ' + str(round(dYmin/self.parameters['dt'], 8)) + ' / ' + str(round(dYmax/self.parameters['dt'], 8)))
		print('wMesh (min/max) = ' + str(round(dZmin/self.parameters['dt'], 8)) + ' / ' + str(round(dZmax/self.parameters['dt'], 8)))

		def minValue(val, a):
			if val > 0:
				val /= a
			else:
				val *= a
			return val

		def maxValue(val, a):
			if val > 0:
				val *= a
			else:
				val /= a
			return val

		c = 1.2
		self.scales = {'minVel': minValue(velMin, c), 'maxVel': maxValue(velMax, c), 'minP': minValue(Pmin, c), 'maxP': maxValue(Pmax, c),
					   'minVelMesh': minValue(velMeshMin, c), 'maxVelMesh': maxValue(velMeshMax, c)}

	def setData(self, mat, nextMat, dataIn, dataOut, B):
		nx, ny, nz, nvar = np.shape(dataIn)

		noise_stddev = 0.05
		u_noise = np.random.normal(loc=0, scale=noise_stddev, size=(nx, ny, nz))
		v_noise = np.random.normal(loc=0, scale=noise_stddev, size=(nx, ny, nz))
		w_noise = np.random.normal(loc=0, scale=noise_stddev, size=(nx, ny, nz))
		p_noise = np.random.normal(loc=0, scale=noise_stddev, size=(nx, ny, nz))

		dataIn[0:nx, 0:ny, 0:nz, 0] = mat['X'][0][0]
		dataIn[0:nx, 0:ny, 0:nz, 1] = mat['Y'][0][0]
		dataIn[0:nx, 0:ny, 0:nz, 2] = mat['Z'][0][0]
		dataIn[0:nx, 0:ny, 0:nz, 3] = (nextMat['X'][0][0] - mat['X'][0][0]) / self.parameters['dt']
		dataIn[0:nx, 0:ny, 0:nz, 4] = (nextMat['Y'][0][0] - mat['Y'][0][0]) / self.parameters['dt']
		dataIn[0:nx, 0:ny, 0:nz, 5] = (nextMat['Z'][0][0] - mat['Z'][0][0]) / self.parameters['dt']
		dataIn[0:nx, 0:ny, 0:nz, 6] = B
		dataIn[0:nx, 0:ny, 0:nz, 7] = mat['D'][0][0]
		dataIn[0:nx, 0:ny, 0:nz, 8] = mat['parameters'][0][0][0][0] / 20
		dataIn[0:nx, 0:ny, 0:nz, 9] = mat['U'][0][0] + u_noise
		dataIn[0:nx, 0:ny, 0:nz, 10] = mat['V'][0][0] + v_noise
		dataIn[0:nx, 0:ny, 0:nz, 11] = mat['W'][0][0] + w_noise
		dataIn[0:nx, 0:ny, 0:nz, 12] = mat['P'][0][0] + p_noise

		dataOut[0:nx, 0:ny, 0:nz, 0] = nextMat['U'][0][0]
		dataOut[0:nx, 0:ny, 0:nz, 1] = nextMat['V'][0][0]
		dataOut[0:nx, 0:ny, 0:nz, 2] = nextMat['W'][0][0]
		dataOut[0:nx, 0:ny, 0:nz, 3] = nextMat['P'][0][0]

	def prepare_training_data(self):
		"""
			mat:    X, Y, B, u, v, p
			dataIn:  X, Y, dx, dy, B, u, v
			dataOut: u, v, p
		"""
		self.setScales()
		with open(self.fileScales, "w") as file:
			json.dump(self.scales, file)

		nSamplesPerDir = 100

		sorted_mat_files = []
		nIter = []
		nIterTot = 0
		for dir in self.dataDirs:
			mat_files = [f for f in Path(dir).iterdir()]
			sorted_mat_files.append(sorted(mat_files, key=lambda filename: int(re.search(r'\d+', filename.name).group())))
			nIter.append(len(sorted_mat_files[-1]))
			nIterTot += nIter[-1]

		mat0 = scipy.io.loadmat(sorted_mat_files[0][0])['data']
		self.nx, self.ny, self.nz = np.shape(mat0['X'][0][0])

		self.nSamplesTot = min(nIterTot, nSamplesPerDir * len(self.dataDirs))
		md = meshDeformation3D(self.parentDir / Path('mesh.mat'))
		B = md.computeB()

		for batch in range(self.nBatches):
			print('batch: ' + str(batch) + ' / ' + str(self.nBatches))
			dataIn = np.zeros((self.batchSize, self.nx, self.ny, self.nz, self.dimIn), dtype=np.float32)
			dataOut = np.zeros((self.batchSize, self.nx, self.ny, self.nz, self.dimOut), dtype=np.float32)

			for iS in range(self.batchSize):
				iDir = random.randint(0, len(self.dataDirs) - 1)
				selected_matIds = random.sample(range(nIter[iDir]-1), min(nSamplesPerDir, self.batchSize))

				for i in selected_matIds:
					print("Creating sample " + str(i) + "/" + str(self.batchSize) + " from " + self.dataDirs[iDir]
					+ " by processing iteration "
					+ re.search(r'\d+', sorted_mat_files[iDir][i].name).group(0) + " and "
					+ re.search(r'\d+', sorted_mat_files[iDir][i + 1].name).group(0))

					mat = scipy.io.loadmat(sorted_mat_files[iDir][i])['data']
					nextmat = scipy.io.loadmat(sorted_mat_files[iDir][i + 1])['data']
					self.setData(mat, nextmat, dataIn[iS], dataOut[iS], B)

			fileIn = self.dataPath / Path('dataIn_' + str(batch) + '.npy')
			fileOut = self.dataPath / Path('dataOut_' + str(batch) + '.npy')
			np.save(fileIn, dataIn)
			np.save(fileOut, dataOut)
