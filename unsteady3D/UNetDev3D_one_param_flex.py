"""U-Net 3D (one parameter) without a fixed input grid size.

Identical architecture, activations, normalization and boundary-condition handling
as UNetDev3D_one_param.UNetDev.  The only difference: the input layer is declared as

    Input(shape=(None, None, None, dimIn))

instead of (n1, n2, n3, dimIn).  The network is fully convolutional
(Conv3D / MaxPooling3D / UpSampling3D with 'same' padding), so nothing in it
depends on the grid size -- only on the number of channels.  A model built this way
can therefore be evaluated on any mesh, e.g. trained on 48x32x16 and run on
96x64x32 or 192x128x64.

n1, n2, n3 are still accepted (and stored) so that the class is a drop-in
replacement for the original one in trainUnet3D*.py, but they are NOT used when
building the graph.

Size requirement
----------------
With `deep` levels the encoder applies (deep - 1) poolings by 2, so every spatial
dimension must be divisible by 2**(deep - 1) (e.g. 16 for deep=5), otherwise the
skip connections do not match after up-sampling.  If `autoPad=True` the model pads
the input up to the next valid multiple (edge replication) and crops the output back,
so that arbitrary sizes work as well.
"""

import tensorflow as tf
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Lambda, Concatenate
from keras.ops import equal, where


def dataNormalization(data, minvalue, maxvalue):
    return (data - minvalue) / (maxvalue - minvalue)


def dataDenormalization(data, minvalue, maxvalue):
    return data * (maxvalue - minvalue) + minvalue


class UNetDev:

    def __init__(self, n1=None, n2=None, n3=None, dimIn=13, dimOut=4, act='relu', actOut='sigmoid',
                 frame_width=1, nChannel=8, deep=5, growFactor=1, scales=None, autoPad=False):
        self.name = "Unet"
        # kept only for compatibility / information -- the graph does not use them
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.act = act
        self.actOut = actOut
        self.frame_width = frame_width
        self.nChannel = nChannel
        self.deep = deep
        self.growFactor = growFactor
        self.scales = scales
        self.autoPad = autoPad
        self.model = None
        self.padding = 'same'

    def info(self):
        print("\n--------------------------------------------------------------")
        print("Model: " + self.name + " -> Unet - version 1 (size independent)")
        print("Required divisor of every spatial dimension: %d" % (2 ** (self.deep - 1)))
        print("--------------------------------------------------------------\n")
        self.model.summary()

    def getChannels(self, nChannel0, deep, growFactor):
        nChannel = []
        for i in range(deep):
            nChannel.append(nChannel0 * i**growFactor)

        return nChannel

    def build(self):
        # parameters
        poolFrame = (2, 2, 2)
        frame = (1 + 2 * self.frame_width, 1 + 2 * self.frame_width, 1 + 2 * self.frame_width)
        nChannels = self.getChannels(self.nChannel, self.deep, self.growFactor)

        # input layer -- the spatial dimensions are left undefined
        input = tf.keras.layers.Input(shape=(None, None, None, self.dimIn))

        x = self.padInput(input) if self.autoPad else input

        input0 = x[..., 0:1]
        input1 = x[..., 1:2]
        input2 = x[..., 2:3]
        input3 = dataNormalization(x[..., 3:4], self.scales['minVelMesh'], self.scales['maxVelMesh'])
        input4 = dataNormalization(x[..., 4:5], self.scales['minVelMesh'], self.scales['maxVelMesh'])
        input5 = dataNormalization(x[..., 5:6], self.scales['minVelMesh'], self.scales['maxVelMesh'])
        input6 = x[..., 6:7]
        input7 = x[..., 7:8]
        input8 = dataNormalization(x[..., 8:9], self.scales['minVel'], self.scales['maxVel'])
        input9 = dataNormalization(x[..., 9:10], self.scales['minVel'], self.scales['maxVel'])
        input10 = dataNormalization(x[..., 10:11], self.scales['minVel'], self.scales['maxVel'])
        input11 = dataNormalization(x[..., 11:12], self.scales['minVel'], self.scales['maxVel'])
        input12 = dataNormalization(x[..., 12:13], self.scales['minP'], self.scales['maxP'])

        uMesh = x[..., 3:4]
        vMesh = x[..., 4:5]
        wMesh = x[..., 5:6]

        layer = Concatenate(axis=4)([input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12])
        # encoder
        conv = [None] * self.deep
        for i in range(1, self.deep):
            conv[i - 1] = Conv3D(nChannels[i], kernel_size=frame, activation=self.act, padding=self.padding)(layer)
            layer = MaxPooling3D(pool_size=poolFrame)(conv[i - 1])

        encoded = Conv3D(self.deep * self.nChannel, kernel_size=frame, activation=self.act, padding=self.padding)(layer)

        # decoder
        layer = encoded
        for i in range(self.deep - 1, 0, -1):
            layer = UpSampling3D(poolFrame)(layer)

            conc = Conv3D(nChannels[i], kernel_size=frame, activation=self.act, padding=self.padding)(layer)
            layer = concatenate([conc, conv[i - 1]])

        out = Conv3D(self.dimOut, kernel_size=frame, activation=self.actOut, padding='same')(layer)

        out0 = dataDenormalization(out[..., 0:1], self.scales['minVel'], self.scales['maxVel'])
        out1 = dataDenormalization(out[..., 1:2], self.scales['minVel'], self.scales['maxVel'])
        out2 = dataDenormalization(out[..., 2:3], self.scales['minVel'], self.scales['maxVel'])
        out3 = dataDenormalization(out[..., 3:4], self.scales['minP'], self.scales['maxP'])
        output = Concatenate(axis=4)([out0, out1, out2, out3])

        output = self.addBC(output, uMesh, vMesh, wMesh, input6)

        if self.autoPad:
            output = self.cropOutput(output, input)

        self.model = tf.keras.models.Model(inputs=input, outputs=output)

    def addBC(self, T, uMesh, vMesh, wMesh, B):
        u = T[..., 0:1]
        v = T[..., 1:2]
        w = T[..., 2:3]
        p = T[..., 3:4]

        # VELOCITY CONDITIONS
        mask = equal(B, 1)
        u = where(mask, uMesh, u)
        v = where(mask, vMesh, v)
        w = where(mask, wMesh, w)

        return Concatenate(axis=4)([u, v, w, p])

    # ------------------------------------------------------------------
    # optional automatic padding to a multiple of 2**(deep-1)
    # ------------------------------------------------------------------
    def padInput(self, x):
        m = 2 ** (self.deep - 1)

        def pad(t):
            shape = tf.shape(t)
            pads = [(m - shape[i] % m) % m for i in (1, 2, 3)]
            paddings = [[0, 0], [0, pads[0]], [0, pads[1]], [0, pads[2]], [0, 0]]
            return tf.pad(t, paddings, mode='SYMMETRIC')

        return Lambda(pad, name='pad_to_multiple')(x)

    def cropOutput(self, y, ref):
        def crop(args):
            t, r = args
            shape = tf.shape(r)
            return t[:, :shape[1], :shape[2], :shape[3], :]

        return Lambda(crop, name='crop_to_input')([y, ref])
