import torch
import torch.nn as nn


class AddBC(nn.Module):
    """
    Vrstva, která přidá okrajové podmínky do 4D tenzoru [nSpec, nx, ny, dims].
    Ve směru nx (osa 2) se opakují krajní hodnoty.
    Ve směru ny (osa 3) se použije periodické kopírování.

    PyTorch očekává vstup v pořadí (batch, channels, height, width).
    Vysvětlení:
    - osa 0: batch
    - osa 1: channels
    - osa 2: height (nx)
    - osa 3: width (ny)
    """

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def forward(self, tensor):
        size = self.size

        left_x = tensor[:, :, 0:1, :].repeat(1, 1, size, 1)
        right_x = tensor[:, :, -1:, :].repeat(1, 1, size, 1)
        padded_x = torch.cat([left_x, tensor, right_x], dim=2)

        left_y = padded_x[:, :, :, -size:]
        right_y = padded_x[:, :, :, :size]
        tout = torch.cat([left_y, padded_x, right_y], dim=3)

        return tout


class UNetDev(nn.Module):
    def __init__(self, n1, n2, dimIn, dimOut,
                 frame_width=1, nChannel=8, deep=5,
                 growFactor=1, scales=None):
        super().__init__()
        self.name = "Unet"
        self.n1 = n1
        self.n2 = n2
        self.dimIn = dimIn
        self.dimOut = dimOut

        self.act = self.nn.ReLU()
        self.actOut = self.nn.Sigmoid()

        self.frame_width = frame_width
        self.nChannel = nChannel
        self.deep = deep
        self.growFactor = growFactor
        self.scales = scales

        self.nChannels = self.getChannels(self.nChannel, self.deep, self.growFactor)
        self.poolFrame = (2, 2)
        self.frame = (1 + 2 * self.frame_width, 1 + 2 * self.frame_width)

        self.model = self.build()


    def getChannels(self, nChannel0, deep, growFactor):
        nChannel = []
        for i in range(deep):
            nChannel.append(int(nChannel0 * (i + 1) ** growFactor))
        return nChannel

    def build(self):
        convs = nn.ModuleList()
        pools = nn.ModuleList()
        upsamples = nn.ModuleList()
        add_bcs = nn.ModuleList()

        # Encoder
        in_channels = self.dimIn
        for i in range(self.deep - 1):
            out_channels = self.nChannels[i]
            add_bcs.append(AddBC(self.frame_width))
            convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.frame, padding='valid'))
            pools.append(nn.MaxPool2d(self.poolFrame))
            in_channels = out_channels

        # Bottleneck
        out_channels_bottleneck = self.deep * self.nChannel
        add_bcs.append(AddBC(self.frame_width))
        convs.append(nn.Conv2d(in_channels, out_channels_bottleneck, kernel_size=self.frame, padding='valid'))

        # Decoder
        in_channels_decoder = out_channels_bottleneck
        for i in range(self.deep - 1, 0, -1):
            out_channels = self.nChannels[i - 1]
            upsamples.append(nn.Upsample(scale_factor=self.poolFrame, mode='bilinear', align_corners=True))
            add_bcs.append(AddBC(self.frame_width))
            convs.append(
                nn.Conv2d(in_channels_decoder + out_channels, out_channels, kernel_size=self.frame, padding='valid'))
            in_channels_decoder = out_channels

        # Output
        add_bcs.append(AddBC(self.frame_width))
        convs.append(nn.Conv2d(in_channels_decoder + self.dimIn, self.dimOut, kernel_size=self.frame, padding='valid'))

        return nn.Sequential(
            *convs, *pools, *upsamples, *add_bcs
        )

    def forward(self, x):
        # Permute input to match PyTorch format: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)

        convs_out = [None] * self.deep
        layer = x

        # Encoder
        for i in range(self.deep - 1):
            layer_bc = self.model[self.deep * 2 + i](layer)  # add_bcs
            convs_out[i] = self.act(self.model[i](layer_bc))  # convs
            layer = self.model[self.deep - 1 + i](convs_out[i])  # pools

        # Bottleneck
        layer_bc = self.model[self.deep * 3 - 2](layer)  # add_bcs
        encoded = self.act(self.model[self.deep - 1](layer_bc))  # convs

        # Decoder
        layer = encoded
        for i in range(self.deep - 1, 0, -1):
            layer = self.model[self.deep * 2 - 2 + (self.deep - 1 - i)](layer)  # upsamples
            layer_bc = self.model[self.deep * 3 - 2 + (self.deep - 1 - i)](layer)  # add_bcs
            conc = self.act(self.model[self.deep + (self.deep - 1 - i)](layer_bc))  # convs
            layer = torch.cat([conc, convs_out[i - 1]], dim=1)

        # Output
        layer_bc = self.model[self.deep * 4 - 3](layer)  # add_bcs
        output = self.model[self.deep + self.deep - 1](layer_bc)  # convs
        if self.actOut is not None:
            output = self.actOut(output)

        # Permute output back to Keras format: (batch, height, width, channels)
        output = output.permute(0, 2, 3, 1)

        return output

    def info(self):
        print("\n--------------------------------------------------------------")
        print("Model: " + self.name + " -> Unet - version 1 (PyTorch)")
        print("--------------------------------------------------------------\n")
        # Výpis modelu v PyTorchi
        print(self)

    def __repr__(self):
        # Custom representation for printing
        return self.__str__()

    def __str__(self):
        # String representation of the model for info()
        return self.model.__str__()