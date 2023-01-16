import torch
from torch import nn


class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Attn(nn.Module):
    def __init__(self, out_channels=[]):
        super().__init__()
        self.conv_1 = nn.Sequential(
            Conv2dIBNormRelu(
                2 * out_channels[0], out_channels[0], 1, stride=1, padding=0),
        )
        self.conv_2 = nn.Sequential(
            Conv2dIBNormRelu(
                2 * out_channels[1], out_channels[1], 1, stride=1, padding=0),
        )
        self.conv_3 = nn.Sequential(
            Conv2dIBNormRelu(
                2 * out_channels[2], out_channels[2], 1, stride=1, padding=0),
        )
        self.conv_4 = nn.Sequential(
            Conv2dIBNormRelu(
                2 * out_channels[3], out_channels[3], 1, stride=1, padding=0),
        )

    def forward_single_frame(self, f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t):
        f1_m = torch.cat((f1, f1_t), 1)
        f2_m = torch.cat((f2, f2_t), 1)
        f3_m = torch.cat((f3, f3_t), 1)
        f4_m = torch.cat((f4, f4_t), 1)
        f1_m = self.conv_1(f1_m)
        f2_m = self.conv_2(f2_m)
        f3_m = self.conv_3(f3_m)
        f4_m = self.conv_4(f4_m)
        return f1_m, f2_m, f3_m, f4_m

    def forward_time_series(self, f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t):
        input_shape = f1.shape
        f1_t = f1_t[:, None, :, :, :]
        f2_t = f2_t[:, None, :, :, :]
        f3_t = f3_t[:, None, :, :, :]
        f4_t = f4_t[:, None, :, :, :]

        f1_t = f1_t.repeat(1, input_shape[1], 1, 1, 1)
        f2_t = f2_t.repeat(1, input_shape[1], 1, 1, 1)
        f3_t = f3_t.repeat(1, input_shape[1], 1, 1, 1)
        f4_t = f4_t.repeat(1, input_shape[1], 1, 1, 1)

        B, T = input_shape[:2]
        f1_m, f2_m, f3_m, f4_m = self.forward_single_frame(
            f1.flatten(0, 1), f2.flatten(0, 1), f3.flatten(0, 1), f4.flatten(0, 1), f1_t.flatten(0, 1), f2_t.flatten(0, 1), f3_t.flatten(0, 1), f4_t.flatten(0, 1))
        f1_m = f1_m.unflatten(0, (B, T))
        f2_m = f2_m.unflatten(0, (B, T))
        f3_m = f3_m.unflatten(0, (B, T))
        f4_m = f4_m.unflatten(0, (B, T))
        return f1_m, f2_m, f3_m, f4_m

    def forward(self, f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t):
        if len(f1.shape) == 5:
            return self.forward_time_series(f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t)
        else:
            return self.forward_single_frame(f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t)

# Global attention module


class GAM(nn.Module):
    def __init__(self, out_channels=[]):
        super().__init__()
        self.conv_1 = nn.Sequential(
            Conv2dIBNormRelu(
                2 * out_channels[0], 2 * out_channels[0], 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                2 * out_channels[0], out_channels[0], 3, stride=1, padding=1),
        )
        self.conv_2 = nn.Sequential(
            Conv2dIBNormRelu(
                2 * out_channels[1], 2 * out_channels[1], 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                2 * out_channels[1], out_channels[1], 3, stride=1, padding=1),
        )
        self.conv_3 = nn.Sequential(
            Conv2dIBNormRelu(
                2 * out_channels[2], 2 * out_channels[2], 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                2 * out_channels[2], out_channels[2], 3, stride=1, padding=1),
        )
        self.conv_4 = nn.Sequential(
            Conv2dIBNormRelu(
                2 * out_channels[3], 2 * out_channels[3], 3, stride=1, padding=1),
            Conv2dIBNormRelu(
                2 * out_channels[3], out_channels[3], 3, stride=1, padding=1),
        )

    def forward_single_frame(self, f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t):
        f1_m = torch.cat((f1, f1_t), 1)
        f2_m = torch.cat((f2, f2_t), 1)
        f3_m = torch.cat((f3, f3_t), 1)
        f4_m = torch.cat((f4, f4_t), 1)
        f1_m = self.conv_1(f1_m)
        f2_m = self.conv_2(f2_m)
        f3_m = self.conv_3(f3_m)
        f4_m = self.conv_4(f4_m)
        return f1_m, f2_m, f3_m, f4_m

    def forward_time_series(self, f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t):
        input_shape = f1.shape
        f1_t = f1_t[:, None, :, :, :]
        f2_t = f2_t[:, None, :, :, :]
        f3_t = f3_t[:, None, :, :, :]
        f4_t = f4_t[:, None, :, :, :]

        f1_t = f1_t.repeat(1, input_shape[1], 1, 1, 1)
        f2_t = f2_t.repeat(1, input_shape[1], 1, 1, 1)
        f3_t = f3_t.repeat(1, input_shape[1], 1, 1, 1)
        f4_t = f4_t.repeat(1, input_shape[1], 1, 1, 1)

        B, T = input_shape[:2]
        f1_m, f2_m, f3_m, f4_m = self.forward_single_frame(
            f1.flatten(0, 1), f2.flatten(0, 1), f3.flatten(0, 1), f4.flatten(0, 1), f1_t.flatten(0, 1), f2_t.flatten(0, 1), f3_t.flatten(0, 1), f4_t.flatten(0, 1))
        f1_m = f1_m.unflatten(0, (B, T))
        f2_m = f2_m.unflatten(0, (B, T))
        f3_m = f3_m.unflatten(0, (B, T))
        f4_m = f4_m.unflatten(0, (B, T))
        return f1_m, f2_m, f3_m, f4_m

    def forward(self, f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t):
        if len(f1.shape) == 5:
            return self.forward_time_series(f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t)
        else:
            return self.forward_single_frame(f1, f2, f3, f4, f1_t, f2_t, f3_t, f4_t)
