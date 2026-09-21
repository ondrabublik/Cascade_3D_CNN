import numpy as np
from pathlib import Path
import random
import scipy.io
import sys
import re
import json
from meshDeformation3D import meshDeformation3D


class Data:
	def __init__(self, dirs, max_ram_gb=None, ram_dtype='auto'):
		self.dataDirs = dirs
		self.parentDir = Path(self.dataDirs[0]).parents[0]
		self.dataPath = self.parentDir / Path('data_3Do')
		self.nSamplesTot = 0
		self.parameters = {}
		self.scales = {}
		self.batchSize = 2
		self.nBatches = 300
		self.dimIn = 13
		self.dimOut = 4
		self.max_ram_gb = max_ram_gb
		self.ram_dtype = ram_dtype
		self._cache_mode = None
		self._in_qmin = None
		self._in_qmax = None
		self._out_qmin = None
		self._out_qmax = None
		self._B = None
		self._param = None

		self.fileParameters = self.dataPath / Path('parameters.json')
		self.fileScales = self.dataPath / Path('scales.json')

		self.dataIn = None
		self.dataOut = None
		self.dataIn_multistep = None
		self.dataOut_multistep = None

		for dir in self.dataDirs:
			if not Path(dir).exists():
				sys.exit("Error: Data directory doesn't exists " + dir)

		self.nx, self.ny, self.nz = self.readGridSize()

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

		self._index_batch_files()
		self.load_all_to_ram()
		self.info()

	def info(self):
		print("\n-------------------------------------")
		print("Data cache mode: ", self._cache_mode)
		if self.dataIn is not None:
			print("Single-step batches: ", len(self.dataIn),
				  "  size: ", round(self._nbytes_gb(self.dataIn, self.dataOut), 3), "GiB")
		if self.dataIn_multistep is not None:
			print("Multistep batches: ", len(self.dataIn_multistep),
				  "  size: ", round(self._nbytes_gb(self.dataIn_multistep, self.dataOut_multistep), 3), "GiB")
		print("-------------------------------------\n")

	@staticmethod
	def _nbytes_gb(*arrays):
		total = 0
		for a in arrays:
			if a is None:
				continue
			if isinstance(a, list):
				total += sum(x.nbytes for x in a)
			else:
				total += a.nbytes
		return total / (1024 ** 3)

	@staticmethod
	def _available_ram_bytes():
		try:
			with open('/proc/meminfo') as f:
				for line in f:
					if line.startswith('MemAvailable:'):
						return int(line.split()[1]) * 1024
		except Exception:
			pass
		try:
			import ctypes
			class MemoryStatusEx(ctypes.Structure):
				_fields_ = [
					('dwLength', ctypes.c_ulong),
					('dwMemoryLoad', ctypes.c_ulong),
					('ullTotalPhys', ctypes.c_ulonglong),
					('ullAvailPhys', ctypes.c_ulonglong),
					('ullTotalPageFile', ctypes.c_ulonglong),
					('ullAvailPageFile', ctypes.c_ulonglong),
					('ullTotalVirtual', ctypes.c_ulonglong),
					('ullAvailVirtual', ctypes.c_ulonglong),
					('ullAvailExtendedVirtual', ctypes.c_ulonglong),
				]
			stat = MemoryStatusEx()
			stat.dwLength = ctypes.sizeof(MemoryStatusEx)
			ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
			return int(stat.ullAvailPhys)
		except Exception:
			return 8 * 1024 ** 3

	def _count_batch_files(self, prefix):
		n = 0
		while (self.dataPath / Path(prefix + '_' + str(n) + '.npy')).exists():
			n += 1
		return n

	def _index_batch_files(self):
		n_single = self._count_batch_files('dataIn')
		n_multi = self._count_batch_files('dataIn_multistep')
		if n_multi > 0:
			self.nBatches = n_multi
			sample = np.load(self.dataPath / Path('dataIn_multistep_0.npy'), mmap_mode='r')
			self.batchSize = sample.shape[0]
		elif n_single > 0:
			self.nBatches = n_single
			sample = np.load(self.dataPath / Path('dataIn_0.npy'), mmap_mode='r')
			self.batchSize = sample.shape[0]
		else:
			print("Error: no batch .npy files found in", self.dataPath)
			exit()

	def _ram_budget_bytes(self, needed_bytes=None):
		if self.max_ram_gb is not None:
			return int(self.max_ram_gb * 1024 ** 3)
		# Leave headroom for TensorFlow / GPU staging.
		return int(self._available_ram_bytes() * 0.75)

	# Packed input channels: drop B (constant mask) and the scalar parameter volume.
	_IN_PACK_IDX = (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12)

	@staticmethod
	def _channel_minmax(arr):
		reduce_axes = tuple(range(arr.ndim - 1))
		vmin = arr.min(axis=reduce_axes).astype(np.float32)
		vmax = arr.max(axis=reduce_axes).astype(np.float32)
		return vmin, vmax

	@staticmethod
	def _to_uint8(arr, vmin, vmax):
		scale = np.divide(255.0, vmax - vmin, out=np.ones_like(vmin), where=(vmax > vmin))
		q = np.clip(np.rint((arr - vmin) * scale), 0, 255)
		return q.astype(np.uint8)

	@staticmethod
	def _from_uint8(q, vmin, vmax):
		return q.astype(np.float32) * ((vmax - vmin) / 255.0) + vmin

	def _pack_input(self, arr):
		"""Keep spatial/flow channels; store B once and parameter as a scalar field."""
		packed = np.take(arr, self._IN_PACK_IDX, axis=-1)
		param = arr[..., 8]
		# parameter is constant in space; keep one value per sample/step
		while param.ndim > arr.ndim - 4:
			param = param[..., 0]
		return packed, param.astype(np.float32)

	def _unpack_input(self, packed, param):
		shape = packed.shape[:-1] + (self.dimIn,)
		out = np.empty(shape, dtype=np.float32)
		out[..., 0:6] = packed[..., 0:6]
		out[..., 6] = self._B
		out[..., 7] = packed[..., 6]
		out[..., 8] = np.reshape(param, packed.shape[:-4] + (1, 1, 1))
		out[..., 9:13] = packed[..., 7:11]
		return out

	def _choose_ram_dtype(self, bytes_f32, bytes_u8):
		choice = self.ram_dtype
		if choice == 'auto':
			budget = self._ram_budget_bytes()
			bytes_f16 = bytes_f32 // 2
			if bytes_f32 <= budget:
				choice = 'float32'
			elif bytes_u8 <= budget:
				choice = 'uint8'
			elif bytes_f16 <= budget:
				choice = 'float16'
			else:
				choice = 'uint8'
		if choice not in ('float32', 'float16', 'uint8'):
			raise ValueError("ram_dtype must be 'auto', 'float32', 'float16' or 'uint8'")
		return choice

	def _store_batch(self, arr, dtype, pack_input=False):
		if pack_input:
			arr, param = self._pack_input(arr)
		else:
			param = None
		if dtype == 'uint8':
			vmin, vmax = self._channel_minmax(arr)
			stored = self._to_uint8(arr, vmin, vmax)
			return stored, vmin, vmax, param
		if dtype == 'float16':
			return arr.astype(np.float16, copy=False), None, None, param
		return np.array(arr, dtype=np.float32, copy=True), None, None, param

	def _restore_batch(self, stored, vmin, vmax, param=None, unpack_input=False):
		if stored.dtype == np.uint8:
			arr = self._from_uint8(stored, vmin, vmax)
		elif stored.dtype == np.float16:
			arr = stored.astype(np.float32)
		else:
			arr = np.asarray(stored, dtype=np.float32)
		if unpack_input and arr.shape[-1] != self.dimIn:
			arr = self._unpack_input(arr, param)
		return arr

	def _load_batch_list(self, prefix_in, prefix_out, n_files):
		sample_in = np.load(self.dataPath / Path(prefix_in + '_0.npy'), mmap_mode='r')
		sample_out = np.load(self.dataPath / Path(prefix_out + '_0.npy'), mmap_mode='r')
		needed_f32 = int(sample_in.nbytes + sample_out.nbytes) * n_files
		bytes_u8_in = int(n_files * sample_in.nbytes * (11 / 13) * 0.25)
		bytes_u8_out = int(n_files * sample_out.nbytes * 0.25)
		bytes_u8 = bytes_u8_in + bytes_u8_out
		dtype = self._choose_ram_dtype(needed_f32, bytes_u8)
		self.ram_dtype = dtype

		print(prefix_in + ' float32: ' + str(round(needed_f32 / 1024 ** 3, 2)) + ' GiB')
		print(prefix_in + ' float16: ' + str(round(needed_f32 / 2 / 1024 ** 3, 2)) + ' GiB')
		print(prefix_in + ' uint8 packed in+out: ' + str(round(bytes_u8 / 1024 ** 3, 2)) + ' GiB'
			  + ' (in ' + str(round(bytes_u8_in / 1024 ** 3, 2))
			  + ' + out ' + str(round(bytes_u8_out / 1024 ** 3, 2)) + ')')
		print('Selected RAM dtype: ' + dtype)

		self._B = np.asarray(sample_in[..., 6], dtype=np.float32)
		while self._B.ndim > 3:
			self._B = self._B[0]

		data_in = [None] * n_files
		data_out = [None] * n_files
		self._in_qmin = [None] * n_files
		self._in_qmax = [None] * n_files
		self._out_qmin = [None] * n_files
		self._out_qmax = [None] * n_files
		self._param = [None] * n_files

		stored_bytes = {'float32': needed_f32, 'float16': needed_f32 // 2, 'uint8': bytes_u8}[dtype]
		budget = self._ram_budget_bytes()
		mmap_out = False
		if stored_bytes > budget and dtype == 'uint8' and bytes_u8_in <= budget:
			mmap_out = True
			stored_bytes = bytes_u8_in
			print('Outputs stay memory-mapped; only inputs are kept in RAM.')
		if stored_bytes > budget:
			self._cache_mode = 'mmap'
			print('Dataset does not fit into RAM; mapping .npy files.')
			for idx in range(n_files):
				data_in[idx] = np.load(self.dataPath / Path(prefix_in + '_' + str(idx) + '.npy'), mmap_mode='r')
				data_out[idx] = np.load(self.dataPath / Path(prefix_out + '_' + str(idx) + '.npy'), mmap_mode='r')
			return data_in, data_out

		self._cache_mode = 'RAM-' + dtype + ('+mmap-out' if mmap_out else '')
		pack = dtype in ('uint8', 'float16')
		for idx in range(n_files):
			if idx % 20 == 0 or idx == n_files - 1:
				print('Loading to RAM (' + dtype + '): ' + str(idx + 1) + ' / ' + str(n_files)
					  + '  (' + prefix_in + ')')
			arr_in = np.load(self.dataPath / Path(prefix_in + '_' + str(idx) + '.npy'))
			data_in[idx], self._in_qmin[idx], self._in_qmax[idx], self._param[idx] = self._store_batch(
				arr_in, dtype, pack_input=pack)
			del arr_in
			if mmap_out:
				data_out[idx] = np.load(self.dataPath / Path(prefix_out + '_' + str(idx) + '.npy'), mmap_mode='r')
			else:
				arr_out = np.load(self.dataPath / Path(prefix_out + '_' + str(idx) + '.npy'))
				data_out[idx], self._out_qmin[idx], self._out_qmax[idx], _ = self._store_batch(
					arr_out, dtype, pack_input=False)
				del arr_out
		return data_in, data_out

	def load_all_to_ram(self):
		n_single = self._count_batch_files('dataIn')
		n_multi = self._count_batch_files('dataIn_multistep')

		# Do not load both datasets: multistep training only needs the multistep files.
		if n_multi > 0 and self.dataIn_multistep is None:
			self.nBatches = n_multi
			self.dataIn_multistep, self.dataOut_multistep = self._load_batch_list(
				'dataIn_multistep', 'dataOut_multistep', n_multi)
			self.batchSize = int(np.load(
				self.dataPath / Path('dataIn_multistep_0.npy'), mmap_mode='r').shape[0])
		elif n_single > 0 and self.dataIn is None:
			self.nBatches = n_single
			self.dataIn, self.dataOut = self._load_batch_list('dataIn', 'dataOut', n_single)
			self.batchSize = int(np.load(
				self.dataPath / Path('dataIn_0.npy'), mmap_mode='r').shape[0])

	def loadDataIn(self, idx):
		return self._restore_batch(
			self.dataIn[idx], self._in_qmin[idx], self._in_qmax[idx],
			param=self._param[idx], unpack_input=True)

	def loadDataOut(self, idx):
		return self._restore_batch(
			self.dataOut[idx], self._out_qmin[idx], self._out_qmax[idx])

	def loadDataIn_multistep(self, idx):
		return self._restore_batch(
			self.dataIn_multistep[idx], self._in_qmin[idx], self._in_qmax[idx],
			param=self._param[idx], unpack_input=True)

	def loadDataOut_multistep(self, idx):
		return self._restore_batch(
			self.dataOut_multistep[idx], self._out_qmin[idx], self._out_qmax[idx])

	def setParameters(self):
		self.parameters = {'Re': 5000, 'dt': 0.1}

	def readGridSize(self):
		mat_files = [f for f in Path(self.dataDirs[0]).iterdir()]
		sorted_mat_files = sorted(mat_files, key=lambda filename: int(re.search(r'\d+', filename.name).group()))
		mat0 = scipy.io.loadmat(sorted_mat_files[0])['data']
		return np.shape(mat0['X'][0][0])

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

		uMesh = (nextMat['X'][0][0] - mat['X'][0][0]) / self.parameters['dt']
		vMesh = (nextMat['Y'][0][0] - mat['Y'][0][0]) / self.parameters['dt']
		wMesh = (nextMat['Z'][0][0] - mat['Z'][0][0]) / self.parameters['dt']

		dataIn[0:nx, 0:ny, 0:nz, 0] = mat['X'][0][0]
		dataIn[0:nx, 0:ny, 0:nz, 1] = mat['Y'][0][0]
		dataIn[0:nx, 0:ny, 0:nz, 2] = mat['Z'][0][0]
		dataIn[0:nx, 0:ny, 0:nz, 3] = uMesh
		dataIn[0:nx, 0:ny, 0:nz, 4] = vMesh
		dataIn[0:nx, 0:ny, 0:nz, 5] = wMesh
		dataIn[0:nx, 0:ny, 0:nz, 6] = B
		dataIn[0:nx, 0:ny, 0:nz, 7] = mat['D'][0][0]
		dataIn[0:nx, 0:ny, 0:nz, 8] = mat['parameters'][0][0][0][0]
		# Na hranici profilu (B==1) rychlost = rychlost site (jako addBC)
		mask = B == 1
		dataIn[0:nx, 0:ny, 0:nz, 9] = np.where(mask, uMesh, mat['U'][0][0] + u_noise)
		dataIn[0:nx, 0:ny, 0:nz, 10] = np.where(mask, vMesh, mat['V'][0][0] + v_noise)
		dataIn[0:nx, 0:ny, 0:nz, 11] = np.where(mask, wMesh, mat['W'][0][0] + w_noise)
		dataIn[0:nx, 0:ny, 0:nz, 12] = mat['P'][0][0] + p_noise

		dataOut[0:nx, 0:ny, 0:nz, 0] = np.where(mask, uMesh, nextMat['U'][0][0])
		dataOut[0:nx, 0:ny, 0:nz, 1] = np.where(mask, vMesh, nextMat['V'][0][0])
		dataOut[0:nx, 0:ny, 0:nz, 2] = np.where(mask, wMesh, nextMat['W'][0][0])
		dataOut[0:nx, 0:ny, 0:nz, 3] = nextMat['P'][0][0]

	def setData_multistep(self, mats, dataIn, dataOut, B):
		"""
		dataIn shape:  [nSteps, nx, ny, nz, dimIn]
		dataOut shape: [nSteps, nx, ny, nz, dimOut]
		"""
		nSteps, nx, ny, nz, _ = np.shape(dataIn)

		for step in range(nSteps):
			mat = mats[step]
			nextMat = mats[step + 1]

			noise_stddev = 0.05
			u_noise = np.random.normal(loc=0, scale=noise_stddev, size=(nx, ny, nz))
			v_noise = np.random.normal(loc=0, scale=noise_stddev, size=(nx, ny, nz))
			w_noise = np.random.normal(loc=0, scale=noise_stddev, size=(nx, ny, nz))
			p_noise = np.random.normal(loc=0, scale=noise_stddev, size=(nx, ny, nz))

			uMesh = (nextMat['X'][0][0] - mat['X'][0][0]) / self.parameters['dt']
			vMesh = (nextMat['Y'][0][0] - mat['Y'][0][0]) / self.parameters['dt']
			wMesh = (nextMat['Z'][0][0] - mat['Z'][0][0]) / self.parameters['dt']

			dataIn[step, :, :, :, 0] = mat['X'][0][0]
			dataIn[step, :, :, :, 1] = mat['Y'][0][0]
			dataIn[step, :, :, :, 2] = mat['Z'][0][0]
			dataIn[step, :, :, :, 3] = uMesh
			dataIn[step, :, :, :, 4] = vMesh
			dataIn[step, :, :, :, 5] = wMesh
			dataIn[step, :, :, :, 6] = B
			dataIn[step, :, :, :, 7] = mat['D'][0][0]
			dataIn[step, :, :, :, 8] = mat['parameters'][0][0][0][0]
			# Na hranici profilu (B==1) rychlost = rychlost site (jako addBC)
			mask = B == 1
			dataIn[step, :, :, :, 9] = np.where(mask, uMesh, mat['U'][0][0] + u_noise)
			dataIn[step, :, :, :, 10] = np.where(mask, vMesh, mat['V'][0][0] + v_noise)
			dataIn[step, :, :, :, 11] = np.where(mask, wMesh, mat['W'][0][0] + w_noise)
			dataIn[step, :, :, :, 12] = mat['P'][0][0] + p_noise

			dataOut[step, :, :, :, 0] = np.where(mask, uMesh, nextMat['U'][0][0])
			dataOut[step, :, :, :, 1] = np.where(mask, vMesh, nextMat['V'][0][0])
			dataOut[step, :, :, :, 2] = np.where(mask, wMesh, nextMat['W'][0][0])
			dataOut[step, :, :, :, 3] = nextMat['P'][0][0]

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

		self.nSamplesTot = min(nIterTot, nSamplesPerDir * len(self.dataDirs))
		md = meshDeformation3D(self.parentDir / Path('mesh.mat'))
		B = md.computeB()

		for batch in range(self.nBatches):
			print('batch: ' + str(batch) + ' / ' + str(self.nBatches))
			dataIn = np.zeros((self.batchSize, self.nx, self.ny, self.nz, self.dimIn), dtype=np.float32)
			dataOut = np.zeros((self.batchSize, self.nx, self.ny, self.nz, self.dimOut), dtype=np.float32)

			for iS in range(self.batchSize):
				iDir = random.randint(0, len(self.dataDirs) - 1)
				i = random.sample(range(nIter[iDir]-1), 1)[0]

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

	def prepare_training_data_multistep(self, nSteps=5):
		"""
			Ulozi sekvence s osou step na zacatku:
			dataIn:  [batch, step, nx, ny, nz, dimIn]
			dataOut: [batch, step, nx, ny, nz, dimOut]
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

		self.nSamplesTot = min(nIterTot, nSamplesPerDir * len(self.dataDirs))
		md = meshDeformation3D(self.parentDir / Path('mesh.mat'))
		B = md.computeB()

		for batch in range(self.nBatches):
			print('batch (multistep): ' + str(batch) + ' / ' + str(self.nBatches))
			dataIn = np.zeros((self.batchSize, nSteps, self.nx, self.ny, self.nz, self.dimIn), dtype=np.float32)
			dataOut = np.zeros((self.batchSize, nSteps, self.nx, self.ny, self.nz, self.dimOut), dtype=np.float32)

			for iS in range(self.batchSize):
				iDir = random.randint(0, len(self.dataDirs) - 1)
				maxStart = max(1, nIter[iDir] - nSteps)
				startCandidates = range(maxStart)
				selectedStartIds = random.sample(startCandidates, min(len(startCandidates), 1))

				for i in selectedStartIds:
					print("Creating multistep sample from " + self.dataDirs[iDir]
					+ " start iteration "
					+ re.search(r'\d+', sorted_mat_files[iDir][i].name).group(0))

					mats = []
					for j in range(nSteps + 1):
						mats.append(scipy.io.loadmat(sorted_mat_files[iDir][i + j])['data'])
					self.setData_multistep(mats, dataIn[iS], dataOut[iS], B)

			fileIn = self.dataPath / Path('dataIn_multistep_' + str(batch) + '.npy')
			fileOut = self.dataPath / Path('dataOut_multistep_' + str(batch) + '.npy')
			np.save(fileIn, dataIn)
			np.save(fileOut, dataOut)
