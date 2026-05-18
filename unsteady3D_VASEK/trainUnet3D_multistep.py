import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.optimizers import Adam
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


class MultiStepDataSequence(tf.keras.utils.Sequence):
    def __init__(self, data, maxDataFiles):
        self.data = data
        self.maxDataFiles = maxDataFiles
        self.nGroups = int(data.nBatches / maxDataFiles)
        self.groupIndex = 0

    def __len__(self):
        return self.maxDataFiles

    def __getitem__(self, idx):
        self.groupIndex += 1
        self.groupIndex = self.groupIndex % self.nGroups
        i = self.groupIndex * self.maxDataFiles + idx
        return self.data.loadDataIn_multistep(i), self.data.loadDataOut_multistep(i)


def build_multistep_rollout_model(base_model, n_steps):
    x_seq = tf.keras.Input(shape=(n_steps,) + tuple(base_model.input_shape[1:]))
    preds = []
    prev_pred = None

    for step in range(n_steps):
        x_step = tf.keras.layers.Lambda(lambda t, i=step: t[:, i, ...])(x_seq)
        geo_step = tf.keras.layers.Lambda(lambda t: t[..., 0:9])(x_step)
        orig_flow_step = tf.keras.layers.Lambda(lambda t: t[..., 9:13])(x_step)

        if step == 0:
            flow_step = orig_flow_step
        else:
            flow_step = prev_pred

        model_input = tf.keras.layers.Concatenate(axis=-1)([geo_step, flow_step])
        prev_pred = base_model(model_input)
        pred_with_step = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, axis=1))(prev_pred)
        preds.append(pred_with_step)

    y_seq = tf.keras.layers.Concatenate(axis=1)(preds)
    return tf.keras.Model(inputs=x_seq, outputs=y_seq, name=f"{base_model.name}_multistep")


def weighted_uvwp_mse(velocity_weight=2.0, pressure_weight=0.5):
    vel_w = tf.constant(velocity_weight, dtype=tf.float32)
    p_w = tf.constant(pressure_weight, dtype=tf.float32)

    def loss(y_true, y_pred):
        # y shape: [batch, step, nx, ny, nz, 4]
        diff_sq = tf.square(y_true - y_pred)
        vel_loss = tf.reduce_mean(diff_sq[..., 0:3])
        p_loss = tf.reduce_mean(diff_sq[..., 3:4])
        return vel_w * vel_loss + p_w * p_loss

    return loss


def trainNetMultistep(
    unet,
    path,
    dataDirs=None,
    epochs=100,
    batch_size=4,
    learningRate=1e-4,
    act="relu",
    actOut="sigmoid",
    frameWidth=2,
    nChannel=16,
    deep=5,
    growFactor=1,
    n_steps=5,
    velocityLossWeight=2.0,
    pressureLossWeight=0.5,
):
    data = Data(dataDirs)
    nx, ny, nz = data.nx, data.ny, data.nz
    dimIn = data.dimIn
    dimOut = data.dimOut
    print(f"nx = {nx}, ny = {ny}, nz = {nz}, n_steps = {n_steps}")

    net = unet(
        nx,
        ny,
        nz,
        dimIn,
        dimOut,
        act=act,
        actOut=actOut,
        scales=data.scales,
        frame_width=frameWidth,
        nChannel=nChannel,
        deep=deep,
        growFactor=growFactor,
    )
    net.build()
    net.info()
    base_model = net.model

    # Ensure multistep datasets exist (step axis is the first axis after batch).
    first_multistep_file = data.dataPath / Path("dataIn_multistep_0.npy")
    if not first_multistep_file.exists():
        data.prepare_training_data_multistep(nSteps=n_steps)

    multistep_model = build_multistep_rollout_model(base_model, n_steps=n_steps)
    optimizer = Adam(learning_rate=learningRate)
    multistep_model.compile(
        loss=weighted_uvwp_mse(
            velocity_weight=velocityLossWeight,
            pressure_weight=pressureLossWeight,
        ),
        optimizer=optimizer,
    )

    path.mkdir(parents=True, exist_ok=True)
    input_params = {
        "modelName": str(path.name),
        "UnetName": str(net.name),
        "dataDirectory": str(dataDirs),
        "epochs": epochs,
        "batch_size": batch_size,
        "learningRate": learningRate,
        "actOut": actOut,
        "frameWidth": frameWidth,
        "nChannel": nChannel,
        "deep": deep,
        "nSteps": n_steps,
        "velocityLossWeight": velocityLossWeight,
        "pressureLossWeight": pressureLossWeight,
    }
    file_path = path / Path("train_params.json")
    with file_path.open("w") as file:
        json.dump(input_params, fp=file, indent=4)

    maxDataFiles = 3
    train_data_sequence = MultiStepDataSequence(data, maxDataFiles)

    history = multistep_model.fit(
        train_data_sequence,
        shuffle=True,
        epochs=epochs,
        verbose=1,
        callbacks=[ErrsEqs(net, path)],
    )

    # Save only base UNet model (without multistep Lambda wrapper).
    base_model.save(path / Path("model.keras"))
    return history


if __name__ == "__main__":
    from UNetDev3D_one_param import UNetDev as Unet
    from dataClass3D_one_param import Data

    import os

    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        print("No GPU devices available.")
    else:
        print("GPU device(s) found:")
        for device in physical_devices:
            print(f"  {device}")

    print("\nPython version: " + sys.version.split()[0])
    print("TensorFlow version: " + tf.__version__)

    # dataDirs = [
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in5_vent10",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in5_vent15",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in5_vent20",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent10",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent15",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent20",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in15_vent10",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in15_vent15",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in15_vent20",
    # ]

    dataDirs = [
        "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed_small/in10_vent10",
        "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed_small/in10_vent15",
        "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed_small/in10_vent20",
    ]
    path = Path("../../data/unet_optuna_v1")

    hist = trainNetMultistep(
        unet=Unet,
        dataDirs=dataDirs,
        epochs=20000,
        batch_size=3,
        frameWidth=2,
        nChannel=26,
        deep=5,
        growFactor=0,
        learningRate=1e-4,
        path=path,
        n_steps=5,
        velocityLossWeight=1.0,
        pressureLossWeight=0.1,
    )

    plotLoss(history=hist, path=path)
    plotErrs(path=path)
