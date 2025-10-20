import sys
import numpy as np
import tensorflow.keras as keras
from keras.optimizers import Adam
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from matplotlib.ticker import FormatStrFormatter


def plotLoss(path, history=None):
    plt.figure()
    for key, val in history.history.items():
        plt.semilogy(history.epoch, val, label=key, linestyle='none'
                     , marker='o', markersize=3, fillstyle='full', alpha=0.5)

    plt.title('Training history')
    plt.legend()
    # plt.show()
    plt.savefig(path / Path('history.png'), dpi=150)


def plotErrs(path):

    file = path / Path("errsHistory.txt")
    if file.exists():
        vals=[]
        with open(file, 'r') as f:
            tokens = [token.replace("[", "").replace("]", "").replace("\n", "") for token in f.readline().split('\t')]
            titles = [title for title in tokens if title]
            for line in f:
                tokens = [token.replace("[", "").replace("]", "").replace("\n", "") for token in
                          line.split('\t')[:-1]]
                vals.append(tokens)

        vals = np.array(vals, dtype='float32')

    fig, axs = plt.subplots(4, 2, figsize=(10, 13))
    fig.tight_layout(h_pad=5, w_pad=5, rect=(0.05, 0.01, 0.98, 0.98))
    # plt.subplots_adjust(left=10, right=10, top=10, bottom=10)

    for i, ax in enumerate(axs.flat):
        if i < len(titles)-1:
            ax.set_title(titles[i+1])
            ax.set(xlabel='epoch')
            ax.xaxis.set_major_formatter(FormatStrFormatter('% d'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('% .2e'))
            ax.set_yscale('log')
            ax.grid()
            ax.plot(vals[:, 0], vals[:, i+1], color='brown', linestyle='none', marker='o'
                , markersize=5,  fillstyle='full')
        else:
            ax.set_axis_off()
    plt.savefig(path / Path('errsHistory.png'), dpi=150)
    plt.close()


class ErrsEqs(keras.callbacks.Callback):
    def __init__(self, net, path):
        super().__init__()
        self.net = net
        self.path = path
        self.errs = dict.fromkeys(
            ["MSE", "Continuity local", "NS local", "Cross-section mass flow"], 0)

        self.file = self.path / Path("errsHistory.txt")
        with open(self.file, 'w') as f:
            f.write('[epoch]\t')
            for key in self.errs.keys():
                f.write('[%s]\t' % key)
            f.write('\n')

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            return

        if epoch % 100 == 0:
            self.net.model.save((self.path / Path("model.keras")))
            # plotErrs(path=self.path)
            plotLoss(path=self.path, history=self.net.model.history)


class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, data, maxDataFiles):
        self.data = data
        self.maxDataFiles = maxDataFiles
        self.nGroups = (int) (data.nBatches / maxDataFiles)
        self.groupIndex = 0

    def __len__(self):
        return self.maxDataFiles

    def __getitem__(self, idx):
        self.groupIndex += 1
        self.groupIndex = self.groupIndex % self.nGroups
        return self.data.loadDataIn(self.groupIndex * self.maxDataFiles + idx), self.data.loadDataOut(self.groupIndex * self.maxDataFiles + idx)


def trainNet(unet, path, dataDirs=None, epochs=100, batch_size=64, learningRate=1e-4, act='relu'
             , actOut='sigmoid', frameWidth=2, nChannel=16, deep=5, growFactor=1, validationSplit=0.1, lossWeights=[]):

    # load train data
    dataIn = np.load(os.path.join(path, "dataIn.npy"))
    dataOut = np.load(os.path.join(path, "dataOut.npy"))
    nSpec, nx, ny, nz, dimIn = np.shape(dataIn)
    nSpec, nx, ny, nz, dimOut = np.shape(dataOut)

    print(f"nx = {nx}, ny = {ny}, nz = {nz}")

    # build a core network model
    net = unet(nx, ny, nz, dimIn, dimOut, act=act, actOut=actOut, frame_width=frameWidth
               , nChannel=nChannel, deep=deep, growFactor=growFactor)

    net.build()
    optimizer = Adam(learning_rate=learningRate)  # Default is 1e-3
    net.model.compile(loss='mean_squared_error', optimizer=optimizer)
    net.info()

    # save parameters
    input_params = {'modelName': str(path.name), 'UnetName': str(net.name),
                    'dataDirectory': str(dataDirs),
                    'epochs': epochs, 'batch_size': batch_size, 'learningRate': learningRate,
                    'actOut': actOut, 'validationSplit': validationSplit,
                    'frameWidth': frameWidth, 'nChannel': nChannel, 'deep': deep,
                    'nSamples': nSpec, 'weights': lossWeights}
    file_path = path / Path('train_params.json')
    path.mkdir(parents=True, exist_ok=True)
    with file_path.open('w') as file:
        json.dump(input_params, fp=file, indent=4)


    # train model
    history = net.model.fit(dataIn, dataOut, shuffle=True, epochs=epochs, verbose=1,
                            callbacks=[ErrsEqs(net, path)])

    # save model
    net.model.save(path / Path("model.keras"))

    return history


if __name__ == "__main__":
    from UNetDev3D_steady2 import UNetDev as Unet

    import os

    os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'


    # Set CUDA_VISIBLE_DEVICES to -1 to disable GPU (optional)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Check available GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("No GPU devices available.")
    else:
        print("GPU device(s) found:")
        for device in physical_devices:
            print(f"  {device}")

    print("\nPyhon version: " + sys.version.split()[0])
    print("TensorFlow version: " + tf.__version__)

    dataDirs = ['../data/training_data/test_3D']
    path = Path('../data/training_data/test_3D')

    hist = trainNet(unet=Unet, dataDirs=dataDirs, epochs=1000, batch_size=5
                    , frameWidth=1, nChannel=12, deep=5, growFactor=1, learningRate=1e-4
                    , path=path, validationSplit=0)

    plotLoss(history=hist, path=path)
    plotErrs(path=path)
