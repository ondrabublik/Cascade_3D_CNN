import tensorflow as tf
from keras.src.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate

class UNetDev:

    def __init__(self, n1, n2, dimIn, dimOut, act='relu', actOut='linear', frame_width=1, nChannel=8, deep=5,
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
            nChannel.append(nChannel0 * i ** growFactor)

        return nChannel

    def build(self):
        # parameters
        poolFrame = (2, 2)
        frame = (1 + 2 * self.frame_width, 1 + 2 * self.frame_width)
        nChannels = self.getChannels(self.nChannel, self.deep, self.growFactor)

        # input layer
        input = tf.keras.layers.Input(shape=(self.n1, self.n2, self.dimIn))

        layer = input
        # encoder
        conv = [None] * self.deep
        for i in range(1, self.deep):
            conv[i - 1] = Conv2D(nChannels[i], kernel_size=frame, activation=self.act, padding='same')(layer)
            layer = MaxPooling2D(pool_size=poolFrame)(conv[i - 1])

        encoded = Conv2D(self.deep * self.nChannel, kernel_size=frame, activation=self.act, padding='same')(layer)

        # decoder
        layer = encoded
        for i in range(self.deep - 1, 0, -1):
            layer = UpSampling2D(poolFrame)(layer)

            conc = Conv2D(nChannels[i], kernel_size=frame, activation=self.act, padding='same')(layer)
            layer = concatenate([conc, conv[i - 1]])

        output = Conv2D(self.dimOut, kernel_size=frame, activation=self.actOut, padding='same')(layer)

        self.model = tf.keras.models.Model(inputs=input, outputs=output)
