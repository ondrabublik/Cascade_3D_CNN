import sys
import numpy as np
import tensorflow.keras as keras
from keras.optimizers import Adam
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf


def plotLoss(path, history=None):
    plt.figure()
    for key, val in history.history.items():
        plt.semilogy(history.epoch, val, label=key, linestyle='none',
                     marker='o', markersize=3, fillstyle='full', alpha=0.5)
    plt.title('Training history')
    plt.legend()
    plt.savefig(path / Path('history.png'), dpi=150)


class myCallback(keras.callbacks.Callback):
    def __init__(self, net, path):
        super().__init__()
        self.net = net
        self.path = path

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            return
        if epoch % 100 == 0:
            self.net.model.save((self.path / Path("model.keras")))
            plotLoss(path=self.path, history=self.net.model.history)


class CustomModel(tf.keras.Model):
    def __init__(self, base_model, scales):
        super().__init__()
        self.base = base_model
        self.scales = scales

    def compile(self, optimizer, metrics=None):
        super().compile()
        self.optimizer = optimizer
        self._metrics = metrics or []

    def train_step(self, data):
        X_batch, y_true = data  # Keras poskytne batch (X,y)

        with tf.GradientTape() as tape:
            y_pred = self.base(X_batch, training=True)

            mse = tf.reduce_mean(tf.square(y_true - y_pred))

            # musíme převést X_batch[...,3] z normalizované škály na fyzickou, pokud používáš fyzický práh
            in_min = self.scales["in_min"]
            in_max = self.scales["in_max"]
            # map normalized -> physical:
            distance_phys = X_batch[..., 0] * (in_max[0] - in_min[0]) + in_min[0]
            mask = tf.cast(distance_phys < 0.0001, tf.float32)

            # target_zero normalized (stejně jako v předchozí části)
            out_min = self.scales["out_min"]; out_max = self.scales["out_max"]
            target_zero = tf.constant([
                (0.0 - out_min[0]) / (out_max[0] - out_min[0]),
                (0.0 - out_min[1]) / (out_max[1] - out_min[1]),
                (0.0 - out_min[2]) / (out_max[2] - out_min[2])
            ], dtype=tf.float32)

            sum_sq = tf.reduce_sum(tf.square(y_pred[..., 0:3] - target_zero), axis=-1)
            penalty = tf.reduce_mean(mask * sum_sq)

            distance_secondary_velocity = X_batch[..., 1] * (in_max[1] - in_min[1]) + in_min[1]
            secondary_velocity = ((X_batch[..., 3] * (in_max[3] - in_min[3]) + in_min[3]) - out_min[2]) / (out_max[2] - out_min[2])
            mask_velo = tf.cast(distance_secondary_velocity < 0.001, tf.float32)
            penalty_secondary_inlet = tf.reduce_mean(mask_velo * tf.square(y_pred[..., 0] - target_zero[0])) \
                      + tf.reduce_mean(mask_velo * tf.square(y_pred[..., 1] - target_zero[1])) \
                      + tf.reduce_mean(mask_velo * tf.square(y_pred[..., 2] - secondary_velocity))

            total_loss = mse + 0.1 * penalty + 1 * penalty_secondary_inlet

        grads = tape.gradient(total_loss, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))

        # Update metrics
        return {"loss": total_loss, "mse": mse, "penalty": penalty, "secondary penalty": penalty_secondary_inlet}



def trainNet(unet, path, dataDirs=None, epochs=100, batch_size=64, learningRate=1e-4,
             act='relu', actOut='sigmoid', frameWidth=2, nChannel=16, deep=5,
             growFactor=1, validationSplit=0.1, lossWeights=[]):

    # load train data
    dataIn = np.load(os.path.join(path, "dataIn.npy"))
    dataOut = np.load(os.path.join(path, "dataOut.npy"))
    scales = np.load(os.path.join(path, "scales.npy"), allow_pickle=True).item()
    nSpec, nx, ny, nz, dimIn = np.shape(dataIn)
    nSpec, nx, ny, nz, dimOut = np.shape(dataOut)

    net = unet(nx, ny, nz, dimIn, dimOut, act=act, actOut=actOut,
               frame_width=frameWidth, nChannel=nChannel, deep=deep, growFactor=growFactor)
    net.build()
    net.info()

    # build a core network model
    base_model = net.model  # tvůj UNet model instance
    custom = CustomModel(base_model, scales)
    custom.compile(optimizer=Adam(1e-4))
    history = custom.fit(dataIn, dataOut, batch_size=batch_size, epochs=epochs)

    net.model.save(path / Path("model.keras"))

    return history


if __name__ == "__main__":
    from UNetDev3D_steady import UNetDev as Unet
    import os

    os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("No GPU devices available.")
    else:
        print("GPU device(s) found:")
        for device in physical_devices:
            print(f"  {device}")

    print("\nPython version: " + sys.version.split()[0])
    print("TensorFlow version: " + tf.__version__)

    dataDirs = ['../data/training_data/test_3D']
    path = Path('../data/training_data/test_3D')

    hist = trainNet(unet=Unet, dataDirs=dataDirs, epochs=50000, batch_size=5,
                    frameWidth=1, nChannel=16, deep=5, growFactor=1, learningRate=1e-4,
                    path=path, validationSplit=0)

    np.save(path / Path("history.npy"), hist.history)
    plotLoss(history=hist, path=path)
