import sys
import numpy as np
import tensorflow.keras as keras
from keras.optimizers import Adam
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from matplotlib.ticker import FormatStrFormatter
import math


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


def smoothExponential(values, alpha=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        smoothed.append(s)
    return smoothed


class LivePlotCallback(keras.callbacks.Callback):
    """Plots training metrics every `plot_every` epochs and saves to disk."""

    def __init__(self, path, plot_every=50):
        super().__init__()
        self.path = path
        self.plot_every = plot_every
        self.epochs_log = []
        self.metrics_log = {}   # key -> list of values

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_log.append(epoch + 1)
        for key, val in logs.items():
            self.metrics_log.setdefault(key, []).append(val)

        if (epoch + 1) % self.plot_every == 0:
            self._save_plot()

    def _save_plot(self):
        epochs = self.epochs_log

        # Separate train vs. val keys
        train_keys = [k for k in self.metrics_log if not k.startswith('val_') and k != 'lr']
        val_keys   = [k for k in self.metrics_log if k.startswith('val_')]
        has_lr     = 'lr' in self.metrics_log

        # Build subplot grid
        n_metric_plots = len(train_keys)   # one panel per metric (train+val overlaid)
        n_extra = 1 if has_lr else 0       # learning-rate panel
        n_panels = n_metric_plots + n_extra

        ncols = 2
        nrows = math.ceil(n_panels / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        fig.suptitle(f'Training progress  (epoch {self.epochs_log[-1]})', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4, w_pad=4)
        axs_flat = list(axs.flat) if hasattr(axs, 'flat') else [axs]

        panel = 0
        for key in train_keys:
            ax = axs_flat[panel]
            y_train = self.metrics_log[key]
            ax.semilogy(epochs, y_train, color='steelblue', alpha=0.35,
                        linestyle='none', marker='o', markersize=3, label=f'train {key}')
            ax.semilogy(epochs, smoothExponential(y_train), color='steelblue',
                        linewidth=1.5, label=f'train {key} (smooth)')

            val_key = 'val_' + key
            if val_key in self.metrics_log:
                y_val = self.metrics_log[val_key]
                ax.semilogy(epochs, y_val, color='tomato', alpha=0.35,
                            linestyle='none', marker='o', markersize=3, label=f'val {key}')
                ax.semilogy(epochs, smoothExponential(y_val), color='tomato',
                            linewidth=1.5, label=f'val {key} (smooth)')

            ax.set_title(key)
            ax.set_xlabel('epoch')
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=7)
            panel += 1

        if has_lr:
            ax = axs_flat[panel]
            ax.semilogy(epochs, self.metrics_log['lr'], color='darkorange',
                        linewidth=1.5, label='learning rate')
            ax.set_title('Learning rate')
            ax.set_xlabel('epoch')
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            ax.legend(fontsize=7)
            panel += 1

        # Hide any unused panels
        for i in range(panel, nrows * ncols):
            axs_flat[i].set_axis_off()

        fig.savefig(self.path / Path('training_progress.png'), dpi=150)
        plt.close(fig)


def dataNormalization(data, minvalue, maxvalue):
    return (data - minvalue) / (maxvalue - minvalue)


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
    def __init__(self, data, maxDataFiles, startBatch=0, nBatches=None):
        self.data = data
        self.maxDataFiles = maxDataFiles
        self.startBatch = startBatch
        total = nBatches if nBatches is not None else data.nBatches - startBatch
        self.nBatches = (total // maxDataFiles) * maxDataFiles  # round down to full groups
        self.nGroups = self.nBatches // maxDataFiles
        self.groupIndex = 0

    def __len__(self):
        return self.maxDataFiles   # same as original: a few steps per epoch

    def __getitem__(self, idx):
        # cycle through groups across epochs
        batch_idx = self.startBatch + self.groupIndex * self.maxDataFiles + idx
        return self.data.loadDataIn(batch_idx), self.data.loadDataOut(batch_idx)

    def on_epoch_end(self):
        self.groupIndex = (self.groupIndex + 1) % max(1, self.nGroups)


def trainNet(unet, path, dataDirs=None, epochs=100, batch_size=64, learningRate=1e-4, act='relu'
             , actOut='sigmoid', frameWidth=2, nChannel=16, deep=5, growFactor=1, validationSplit=0.1, lossWeights=[]):

    # load train data
    data = Data(dataDirs)
    nx = data.nx
    ny = data.ny
    nz = data.nz
    print(f"nx = {nx}, ny = {ny}, nz = {nz}")
    dimIn = data.dimIn
    dimOut = data.dimOut

    # build a core network model
    net = unet(nx, ny, nz, dimIn, dimOut, act=act, actOut=actOut, scales=data.scales, frame_width=frameWidth
               , nChannel=nChannel, deep=deep, growFactor=growFactor)

    net.build()
    optimizer = Adam(learning_rate=learningRate)  # Default is 1e-3
    net.model.compile(loss='mean_squared_error', optimizer=optimizer,
                      metrics=['mae'])  # MAE logged alongside MSE loss
    net.info()

    # save parameters
    input_params = {'modelName': str(path.name), 'UnetName': str(net.name),
                    'dataDirectory': str(dataDirs),
                    'epochs': epochs, 'batch_size': batch_size, 'learningRate': learningRate,
                    'actOut': actOut, 'validationSplit': validationSplit,
                    'frameWidth': frameWidth, 'nChannel': nChannel, 'deep': deep,
                    'nSamples': data.nBatches * data.batchSize, 'weights': lossWeights}
    file_path = path / Path('train_params.json')
    path.mkdir(parents=True, exist_ok=True)
    with file_path.open('w') as file:
        json.dump(input_params, fp=file, indent=4)

    # Define data sequence — split into train / val
    maxDataFiles = 3
    total_batches = data.nBatches
    n_val_batches  = max(maxDataFiles, int(total_batches * validationSplit // maxDataFiles) * maxDataFiles)
    n_train_batches = total_batches - n_val_batches

    train_data_sequence = DataSequence(data, maxDataFiles, startBatch=0,           nBatches=n_train_batches)
    val_data_sequence   = DataSequence(data, maxDataFiles, startBatch=n_train_batches, nBatches=n_val_batches) \
                          if validationSplit > 0 and n_val_batches > 0 else None

    print(f"Train batches: {n_train_batches}  |  Val batches: {n_val_batches}  |  Total: {total_batches}")

    # train model
    live_plot  = LivePlotCallback(path=path, plot_every=50)
    lr_logger  = keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr, verbose=0)

    history = net.model.fit(train_data_sequence, shuffle=True, epochs=epochs, verbose=1,
                            validation_data=val_data_sequence,
                            callbacks=[ErrsEqs(net, path), live_plot, lr_logger])

    # save model
    net.model.save(path / Path("model.keras"))

    return history


if __name__ == "__main__":
    from dataClass3D import Data
    from UNetDev3D import UNetDev as Unet

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

    # dataDirs = ['../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in5_vent10',
    #             '../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in5_vent15',
    #             '../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in5_vent20',
    #             '../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent10',
    #             '../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent15',
    #             '../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent20',
    #             '../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in15_vent10',
    #             '../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in15_vent15',
    #             '../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in15_vent20']
    path = Path('../MODELS/unet3D_small')
    dataDirs = [
        "../DATA/transformed_small/in15_vent10",
        "../DATA/transformed_small/in15_vent15",
        "../DATA/transformed_small/in15_vent20"
    ]

    hist = trainNet(unet=Unet, dataDirs=dataDirs, epochs=20000, batch_size=5
                    , frameWidth=2, nChannel=16, deep=4, growFactor=1, learningRate=1e-4
                    , path=path, validationSplit=0.1)

    plotLoss(history=hist, path=path)
    plotErrs(path=path)
