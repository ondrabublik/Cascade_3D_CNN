import tensorflow as tf
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Lambda, Concatenate
from keras.ops import equal, where

def dataNormalization(data, minvalue, maxvalue):
    return (data - minvalue) / (maxvalue - minvalue)


def dataDenormalization(data, minvalue, maxvalue):
    return data * (maxvalue - minvalue) + minvalue


class UNetDev:

    def __init__(self, n1, n2, n3, dimIn, dimOut, act='relu', actOut='sigmoid', frame_width=1, nChannel=8, deep=5, growFactor=1,  scales=None):
        self.name = "Unet"
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
        self.model = None
        self.padding = 'same'

    def info(self):
        print("\n--------------------------------------------------------------")
        print("Model: " + self.name + " -> Unet - version 1")
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

        # input layer
        input = tf.keras.layers.Input(shape=(self.n1, self.n2, self.n3, self.dimIn))

        input0 = input[..., 0:1]
        input1 = input[..., 1:2]
        input2 = input[..., 2:3]
        input3 = dataNormalization(input[..., 3:4], self.scales['minVelMesh'], self.scales['maxVelMesh'])
        input4 = dataNormalization(input[..., 4:5], self.scales['minVelMesh'], self.scales['maxVelMesh'])
        input5 = dataNormalization(input[..., 5:6], self.scales['minVelMesh'], self.scales['maxVelMesh'])
        input6 = input[..., 6:7]
        input7 = input[..., 7:8]
        input8 = input[..., 8:9]
        input9 = dataNormalization(input[..., 9:10], self.scales['minVel'], self.scales['maxVel'])
        input10 = dataNormalization(input[..., 10:11], self.scales['minVel'], self.scales['maxVel'])
        input11 = dataNormalization(input[..., 11:12], self.scales['minVel'], self.scales['maxVel'])
        input12 = dataNormalization(input[..., 12:13], self.scales['minP'], self.scales['maxP'])

        uMesh = input[..., 3:4]
        vMesh = input[..., 4:5]
        wMesh = input[..., 5:6]

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
