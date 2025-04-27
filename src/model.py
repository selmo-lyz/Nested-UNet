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
    def __init__(self, layer_configs=None):
        super().__init__()

        if layer_configs is None:
            layer_configs = self._generate_default_layer_configs()

        self.nodes = nn.ModuleDict()

        for name, config in layer_configs["EncoderNode"].items():
            self.nodes[name] = EncoderNode(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                sampling_method=config["sampling_method"],
            )

        for name, config in layer_configs["DecoderNode"].items():
            self.nodes[name] = DecoderNode(
                in_channels=config["in_channels"],
                upsampling_in_channels=config["upsampling_in_channels"],
                out_channels=config["out_channels"],
                sampling_method=config["sampling_method"],
            )

        deepsupervision_config = layer_configs.get("DeepSupervisionModule", None)
        if deepsupervision_config is not None:
            config = deepsupervision_config["deep_supervision"]
            self.nodes["deep_supervision"] = DeepSupervisionModule(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                num_levels=config["num_level"],
            )

    def forward(self, input):
        outputs = {}
        inputs = {}

        # NestedUNet: Depth 0
        outputs["node0_0"], inputs["node1_0"] = self.nodes["node0_0"](input)
        # NestedUNet: Depth 1 ~ 4
        for depth in range(1, 5):
            # Encoder
            enc_name = f"node{depth}_0"
            outputs[enc_name], inputs[f"node{depth+1}_0"] = self.nodes[enc_name](
                inputs[enc_name]
            )

            # Decoder
            for level in range(1, depth + 1):
                dec_name = f"node{depth - level}_{level}"
                skip_inputs = [
                    outputs[f"node{depth - level}_{k}"] for k in range(level)
                ]
                outputs[dec_name] = self.nodes[dec_name](
                    outputs[f"node{depth - level + 1}_{level - 1}"], skip_inputs
                )

        # NestedUNet: Deep Supervision
        output = [outputs[f"node0_{level}"] for level in range(1, 5)]
        if "deep_supervision" in self.nodes:
            output = self.nodes["deep_supervision"](output)

        return output

    def _generate_default_layer_configs(self):
        encoder_channels = [32, 64, 128, 256, 512]
        layer_configs = {
            "EncoderNode": {
                "node0_0": {
                    "in_channels": 1,
                    "out_channels": encoder_channels[0],
                    "sampling_method": "pool",
                },
                "node1_0": {
                    "in_channels": encoder_channels[0],
                    "out_channels": encoder_channels[1],
                    "sampling_method": "pool",
                },
                "node2_0": {
                    "in_channels": encoder_channels[1],
                    "out_channels": encoder_channels[2],
                    "sampling_method": "pool",
                },
                "node3_0": {
                    "in_channels": encoder_channels[2],
                    "out_channels": encoder_channels[3],
                    "sampling_method": "pool",
                },
                "node4_0": {
                    "in_channels": encoder_channels[3],
                    "out_channels": encoder_channels[4],
                    "sampling_method": "none",
                },
            },
            "DecoderNode": {
                # NestedUNet: Depth 1 Decoder
                "node0_1": {
                    "in_channels": encoder_channels[0] + encoder_channels[1],
                    "upsampling_in_channels": encoder_channels[1],
                    "out_channels": encoder_channels[0],
                    "sampling_method": "upsample",
                },
                # NestedUNet: Depth 2 Decoder
                "node1_1": {
                    "in_channels": encoder_channels[1] + encoder_channels[2],
                    "upsampling_in_channels": encoder_channels[2],
                    "out_channels": encoder_channels[1],
                    "sampling_method": "upsample",
                },
                "node0_2": {
                    "in_channels": encoder_channels[0] * 2 + encoder_channels[1],
                    "upsampling_in_channels": encoder_channels[1],
                    "out_channels": encoder_channels[0],
                    "sampling_method": "upsample",
                },
                # NestedUNet: Depth 3 Decoder
                "node2_1": {
                    "in_channels": encoder_channels[2] + encoder_channels[3],
                    "upsampling_in_channels": encoder_channels[3],
                    "out_channels": encoder_channels[2],
                    "sampling_method": "upsample",
                },
                "node1_2": {
                    "in_channels": encoder_channels[1] * 2 + encoder_channels[2],
                    "upsampling_in_channels": encoder_channels[2],
                    "out_channels": encoder_channels[1],
                    "sampling_method": "upsample",
                },
                "node0_3": {
                    "in_channels": encoder_channels[0] * 3 + encoder_channels[1],
                    "upsampling_in_channels": encoder_channels[1],
                    "out_channels": encoder_channels[0],
                    "sampling_method": "upsample",
                },
                # NestedUNet: Depth 4 Decoder
                "node3_1": {
                    "in_channels": encoder_channels[3] + encoder_channels[4],
                    "upsampling_in_channels": encoder_channels[4],
                    "out_channels": encoder_channels[3],
                    "sampling_method": "upsample",
                },
                "node2_2": {
                    "in_channels": encoder_channels[2] * 2 + encoder_channels[3],
                    "upsampling_in_channels": encoder_channels[3],
                    "out_channels": encoder_channels[2],
                    "sampling_method": "upsample",
                },
                "node1_3": {
                    "in_channels": encoder_channels[1] * 3 + encoder_channels[2],
                    "upsampling_in_channels": encoder_channels[2],
                    "out_channels": encoder_channels[1],
                    "sampling_method": "upsample",
                },
                "node0_4": {
                    "in_channels": encoder_channels[0] * 4 + encoder_channels[1],
                    "upsampling_in_channels": encoder_channels[1],
                    "out_channels": encoder_channels[0],
                    "sampling_method": "upsample",
                },
            },
            "DeepSupervisionModule": {
                "deep_supervision": {
                    "in_channels": encoder_channels[0],
                    "out_channels": 1,
                    "num_level": 4,
                },
            },
        }

        return layer_configs
