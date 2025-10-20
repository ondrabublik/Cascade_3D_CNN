import tensorflow as tf
from keras.src.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from keras.src.models import Model


class AddBC(tf.keras.layers.Layer):
    """
    Vrstva, která přidá okrajové podmínky do 4D tenzoru [nSpec, nx, ny, dims].
    Ve směru nx (osa 1) se opakují krajní hodnoty.
    Ve směru ny (osa 2) se použije periodické kopírování.
    """
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def call(self, tensor):
        size = self.size

        # --- směr x (osa 1): opakujeme krajní hodnoty ---
        left_x = tf.repeat(tensor[:, 0:1, :, :], size, axis=1)
        right_x = tf.repeat(tensor[:, -1:, :, :], size, axis=1)
        padded_x = tf.concat([left_x, tensor, right_x], axis=1)

        # --- směr y (osa 2): periodicita ---
        left_y = padded_x[:, :, -size:, :]
        right_y = padded_x[:, :, :size, :]
        tout = tf.concat([left_y, padded_x, right_y], axis=2)

        return tout


class UNetDev:

    def __init__(self, n1, n2, dimIn, dimOut,
                 act='relu', actOut='linear',
                 frame_width=1, nChannel=8, deep=5,
                 growFactor=1, scales=None):
        self.name = "Unet"
        self.n1 = n1
        self.n2 = n2
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.act = act
        self.actOut = actOut
        self.frame_width = frame_width
        self.nChannel = nChannel
        self.deep = deep
        self.growFactor = growFactor
        self.scales = scales
        self.model = None

    def info(self):
        print("\n--------------------------------------------------------------")
        print("Model: " + self.name + " -> Unet - version 1")
        print("--------------------------------------------------------------\n")
        self.model.summary()

    def getChannels(self, nChannel0, deep, growFactor):
        nChannel = []
        for i in range(deep):
            nChannel.append(nChannel0 * (i + 1) ** growFactor)
        return nChannel

    def build(self):
        poolFrame = (2, 2)
        frame = (1 + 2 * self.frame_width, 1 + 2 * self.frame_width)
        nChannels = self.getChannels(self.nChannel, self.deep, self.growFactor)

        # input layer
        inputs = Input(shape=(self.n1, self.n2, self.dimIn))

        layer = inputs
        conv = [None] * self.deep

        # encoder
        for i in range(1, self.deep):
            layer_bc = AddBC(self.frame_width)(layer)
            conv[i - 1] = Conv2D(nChannels[i], kernel_size=frame,
                                 activation=self.act, padding='valid')(layer_bc)
            layer = MaxPooling2D(pool_size=poolFrame)(conv[i - 1])

        # bottleneck
        layer_bc = AddBC(self.frame_width)(layer)
        encoded = Conv2D(self.deep * self.nChannel, kernel_size=frame, activation=self.act, padding='valid')(layer_bc)

        # decoder
        layer = encoded
        for i in range(self.deep - 1, 0, -1):
            layer = UpSampling2D(poolFrame)(layer)
            layer_bc = AddBC(self.frame_width)(layer)
            conc = Conv2D(nChannels[i], kernel_size=frame, activation=self.act, padding='valid')(layer_bc)
            layer = concatenate([conc, conv[i - 1]])

        # output
        layer_bc = AddBC(self.frame_width)(layer)
        output = Conv2D(self.dimOut, kernel_size=frame,
                        activation=self.actOut, padding='valid')(layer_bc)

        self.model = Model(inputs=inputs, outputs=output)
