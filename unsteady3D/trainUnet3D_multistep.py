import json
import math
import sys
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported. The interactive Tk backend is not thread-safe:
# Keras loads data in worker threads and a figure garbage-collected off the main thread
# kills the process with "Tcl_AsyncDelete: async handler deleted by the wrong thread".
# This script only ever writes PNGs, so the non-interactive backend is the right one.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.optimizers import Adam
from matplotlib.ticker import FormatStrFormatter


def plotLoss(path, history=None):
    plt.figure()
    for key, val in history.history.items():
        plt.semilogy(
            history.epoch,
            val,
            label=key,
            linestyle="none",
            marker="o",
            markersize=3,
            fillstyle="full",
            alpha=0.5,
        )

    plt.title("Training history")
    plt.legend()
    plt.savefig(path / Path("history.png"), dpi=150)


def plotErrs(path):
    file = path / Path("errsHistory.txt")
    if file.exists():
        vals = []
        with open(file, "r") as f:
            tokens = [
                token.replace("[", "").replace("]", "").replace("\n", "")
                for token in f.readline().split("\t")
            ]
            titles = [title for title in tokens if title]
            for line in f:
                tokens = [
                    token.replace("[", "").replace("]", "").replace("\n", "")
                    for token in line.split("\t")[:-1]
                ]
                vals.append(tokens)

        vals = np.array(vals, dtype="float32")

    fig, axs = plt.subplots(4, 2, figsize=(10, 13))
    fig.tight_layout(h_pad=5, w_pad=5, rect=(0.05, 0.01, 0.98, 0.98))

    for i, ax in enumerate(axs.flat):
        if i < len(titles) - 1:
            ax.set_title(titles[i + 1])
            ax.set(xlabel="epoch")
            ax.xaxis.set_major_formatter(FormatStrFormatter("% d"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("% .2e"))
            ax.set_yscale("log")
            ax.grid()
            ax.plot(
                vals[:, 0],
                vals[:, i + 1],
                color="brown",
                linestyle="none",
                marker="o",
                markersize=5,
                fillstyle="full",
            )
        else:
            ax.set_axis_off()
    plt.savefig(path / Path("errsHistory.png"), dpi=150)
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
    """Logs every Keras metric each epoch and plots them every `plot_every` epochs.

    The text log is the authoritative record: it is appended and closed after every
    single epoch, so it survives an interrupted run. The format is the tab-separated
    one plotErrs() already reads, which is also plain TSV for any external tool.
    """

    def __init__(self, path, plot_every=50, logName='errsHistory.txt'):
        super().__init__()
        self.path = path
        self.plot_every = plot_every
        self.logFile = self.path / Path(logName)
        self.columns = None     # fixed on the first epoch, once the metric keys are known
        self.epochs_log = []
        self.metrics_log = {}   # key -> list of values

    def _writeLogRow(self, epoch, logs):
        if self.columns is None:
            self.columns = list(logs.keys())
            with open(self.logFile, 'w') as f:
                f.write('[epoch]\t')
                for key in self.columns:
                    f.write('[%s]\t' % key)
                f.write('\n')

        with open(self.logFile, 'a') as f:
            f.write('%d\t' % epoch)
            for key in self.columns:
                f.write('%.6e\t' % logs.get(key, float('nan')))
            f.write('\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_log.append(epoch + 1)
        for key, val in logs.items():
            self.metrics_log.setdefault(key, []).append(val)

        self._writeLogRow(epoch + 1, logs)

        if (epoch + 1) % self.plot_every == 0:
            self._save_plot()

    def _save_plot(self):
        epochs = self.epochs_log

        # Separate train vs. val keys
        train_keys = [k for k in self.metrics_log if not k.startswith('val_')]

        # Build subplot grid: one panel per metric (train+val overlaid)
        n_panels = len(train_keys)

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

        # Hide any unused panels
        for i in range(panel, nrows * ncols):
            axs_flat[i].set_axis_off()

        fig.savefig(self.path / Path('training_progress.png'), dpi=150)
        plt.close(fig)


class ErrsEqs(keras.callbacks.Callback):
    """Checkpoints the base UNet: 'model.keras' every 10 epochs (latest state) and
    'model_best.keras' whenever `monitor` improves.

    Keras' own ModelCheckpoint is not usable here: fit() runs the unrolled rollout
    wrapper, so it would save that instead of the plain UNet the inference scripts
    expect. Metric logging lives in LivePlotCallback.
    """

    def __init__(self, net, path, monitor="val_loss"):
        super().__init__()
        self.net = net
        self.path = path
        self.monitor = monitor
        self.best = float("inf")

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            return

        if epoch % 10 == 0:
            self.net.model.save((self.path / Path("model.keras")))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is not None and current < self.best:
            self.best = current
            self.net.model.save((self.path / Path("model_best.keras")))


class MultiStepDataSequence(tf.keras.utils.Sequence):
    """One item = one .npy batch file, one epoch = every file in fileIds.

    fileIds is explicit so the caller can hold a disjoint set of files back for
    validation.
    """

    def __init__(self, data, fileIds, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.fileIds = list(fileIds)

    def __len__(self):
        return len(self.fileIds)

    def __getitem__(self, idx):
        i = self.fileIds[idx]
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
    validationSplit=0.2,
    clipNorm=1.0,
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
    # clipnorm guards the rollout: the gradient runs through n_steps chained calls of
    # the same UNet, where one bad sequence can otherwise produce a huge update.
    optimizer = Adam(learning_rate=learningRate, clipnorm=clipNorm)
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
        "validationSplit": validationSplit,
        "clipNorm": clipNorm,
    }
    file_path = path / Path("train_params.json")
    with file_path.open("w") as file:
        json.dump(input_params, fp=file, indent=4)

    # Hold the last files back for validation. fit() cannot do validation_split on a
    # Sequence -- it only splits arrays it can slice -- so the split has to be explicit.
    nVal = int(round(data.nBatches * validationSplit))
    nVal = min(nVal, data.nBatches - 1)
    trainIds = range(data.nBatches - nVal)
    valIds = range(data.nBatches - nVal, data.nBatches)

    train_data_sequence = MultiStepDataSequence(data, trainIds)
    val_data_sequence = MultiStepDataSequence(data, valIds) if nVal else None
    print(f"train files: {len(trainIds)}, validation files: {nVal}")

    # Without a validation split there is no val_loss to watch, so fall back to loss.
    monitorKey = "val_loss" if val_data_sequence else "loss"
    print(f"monitoring '{monitorKey}' for the best checkpoint")

    live_plot = LivePlotCallback(path=path, plot_every=10)

    history = multistep_model.fit(
        train_data_sequence,
        validation_data=val_data_sequence,
        shuffle=True,
        epochs=epochs,
        verbose=1,
        callbacks=[ErrsEqs(net, path, monitor=monitorKey), live_plot],
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

    # dataDirs = [
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent10",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent15",
    #     "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed/in10_vent20",
    # ]

    # Anchor paths to this file, not to the current working directory.
    projectDir = Path(__file__).resolve().parents[1]

    dataDirs = [
        str(projectDir / "DATA" / "data_small" / "transformed_10")
    ]

    path = projectDir / "data" / "net6_3D_multistep_full"

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
