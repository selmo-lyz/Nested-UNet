import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.seq_modules = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.seq_modules(x)
        return x


class EncoderNode(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        sampling_method="conv",
    ):
        super().__init__()
        self.convs = ConvModule(in_channels, out_channels, kernel_size, padding)
        self.downsampling = nn.Identity()

        if sampling_method == "conv":
            self.downsampling = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            )
        elif sampling_method == "pool":
            self.downsampling = nn.MaxPool2d(kernel_size=2)
        elif sampling_method == "none":
            self.downsampling = nn.Identity()
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    def forward(self, x):
        x = self.convs(x)
        x_downsampled = self.downsampling(x)
        return x, x_downsampled


class DecoderNode(nn.Module):
    def __init__(
        self,
        in_channels,
        upsampling_in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        sampling_method="conv",
    ):
        super().__init__()
        self.convs = ConvModule(in_channels, out_channels, kernel_size, padding)
        self.upsampling = nn.Identity()

        if sampling_method == "conv":
            self.upsampling = nn.ConvTranspose2d(
                upsampling_in_channels,
                upsampling_in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        elif sampling_method == "upsample":
            self.upsampling = nn.Upsample(scale_factor=2, mode="bilinear")
        elif sampling_method == "none":
            self.upsampling = nn.Identity()
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    def forward(self, x, skip=None):
        x = self.upsampling(x)
        if skip is not None:
            x = torch.concat(skip + [x], 1)
        x = self.convs(x)
        return x


class DeepSupervisionModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels=4):
        super().__init__()
        self.segs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=True,
                )
                for i in range(num_levels)
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs=[]):
        for i, x in enumerate(xs):
            xs[i] = self.sigmoid(self.segs[i](x))
        return xs


class NestedUNet(nn.Module):
    def __init__(self, encoder_channels=[32, 64, 128, 256, 512]):
        super().__init__()
        """
        NestedUNet: Encoder
        """
        self.node0_0 = EncoderNode(1, encoder_channels[0], sampling_method="pool")
        self.node1_0 = EncoderNode(
            encoder_channels[0], encoder_channels[1], sampling_method="pool"
        )
        self.node2_0 = EncoderNode(
            encoder_channels[1], encoder_channels[2], sampling_method="pool"
        )
        self.node3_0 = EncoderNode(
            encoder_channels[2], encoder_channels[3], sampling_method="pool"
        )
        self.node4_0 = EncoderNode(
            encoder_channels[3], encoder_channels[4], sampling_method="none"
        )
        """
        NestedUNet: Level 1 Decoder
        """
        self.node0_1 = DecoderNode(
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[1],
            encoder_channels[0],
            sampling_method="upsample",
        )
        """
        NestedUNet: Level 2 Decoder
        """
        self.node1_1 = DecoderNode(
            encoder_channels[1] + encoder_channels[2],
            encoder_channels[2],
            encoder_channels[1],
            sampling_method="upsample",
        )
        self.node0_2 = DecoderNode(
            encoder_channels[0] * 2 + encoder_channels[1],
            encoder_channels[1],
            encoder_channels[0],
            sampling_method="upsample",
        )
        """
        NestedUNet: Level 3 Decoder
        """
        self.node2_1 = DecoderNode(
            encoder_channels[2] + encoder_channels[3],
            encoder_channels[3],
            encoder_channels[2],
            sampling_method="upsample",
        )
        self.node1_2 = DecoderNode(
            encoder_channels[1] * 2 + encoder_channels[2],
            encoder_channels[2],
            encoder_channels[1],
            sampling_method="upsample",
        )
        self.node0_3 = DecoderNode(
            encoder_channels[0] * 3 + encoder_channels[1],
            encoder_channels[1],
            encoder_channels[0],
            sampling_method="upsample",
        )
        """
        NestedUNet: Level 4 Decoder
        """
        self.node3_1 = DecoderNode(
            encoder_channels[3] + encoder_channels[4],
            encoder_channels[4],
            encoder_channels[3],
            sampling_method="upsample",
        )
        self.node2_2 = DecoderNode(
            encoder_channels[2] * 2 + encoder_channels[3],
            encoder_channels[3],
            encoder_channels[2],
            sampling_method="upsample",
        )
        self.node1_3 = DecoderNode(
            encoder_channels[1] * 3 + encoder_channels[2],
            encoder_channels[2],
            encoder_channels[1],
            sampling_method="upsample",
        )
        self.node0_4 = DecoderNode(
            encoder_channels[0] * 4 + encoder_channels[1],
            encoder_channels[1],
            encoder_channels[0],
            sampling_method="upsample",
        )
        """
        NestedUNet: Deep Supervision
        """
        self.deep_supervision = DeepSupervisionModule(encoder_channels[0], 1)

    def forward(self, input):
        """
        NestedUNet: Level 1
        """
        x0_0, x0_0_downsampled = self.node0_0(input)
        x1_0, x1_0_downsampled = self.node1_0(x0_0_downsampled)
        x0_1 = self.node0_1(x1_0, [x0_0])
        """
        NestedUNet: Level 2
        """
        x2_0, x2_0_downsampled = self.node2_0(x1_0_downsampled)
        x1_1 = self.node1_1(x2_0, [x1_0])
        x0_2 = self.node0_2(x1_1, [x0_0, x0_1])
        """
        NestedUNet: Level 3
        """
        x3_0, x3_0_downsampled = self.node3_0(x2_0_downsampled)
        x2_1 = self.node2_1(x3_0, [x2_0])
        x1_2 = self.node1_2(x2_1, [x1_0, x1_1])
        x0_3 = self.node0_3(x1_2, [x0_0, x0_1, x0_2])
        """
        NestedUNet: Level 4
        """
        x4_0, _ = self.node4_0(x3_0_downsampled)
        x3_1 = self.node3_1(x4_0, [x3_0])
        x2_2 = self.node2_2(x3_1, [x2_0, x2_1])
        x1_3 = self.node1_3(x2_2, [x1_0, x1_1, x1_2])
        x0_4 = self.node0_4(x1_3, [x0_0, x0_1, x0_2, x0_3])
        """
        NestedUNet: Deep Supervision
        """
        outputs = self.deep_supervision([x0_1, x0_2, x0_3, x0_4])

        return outputs
