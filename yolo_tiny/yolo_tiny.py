import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=2, pool_stride=2, pool_padding=0):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, 3, 1, 1),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
        )

    def forward(self, x):
        return self.block(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, up_scale=2):
        super(Upsample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * up_scale ** 2, kernel_size=3, stride=1,
                      padding=1),
            nn.PixelShuffle(upscale_factor=up_scale),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class YoloTiny(nn.Module):
    def __init__(self):
        super(YoloTiny, self).__init__()
        self.backbone26 = nn.Sequential(
            Block(3, 16),
            Block(16, 32),
            Block(32, 64),
            Block(64, 128),
            ConvBlock(128, 256, 3, 1, 1),
        )

        self.backbone13 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            Block(256, 512, pool_kernel=3, pool_stride=1, pool_padding=1),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 256, 1, 1)
        )

        self.layer13 = nn.Sequential(
            ConvBlock(256, 512, 3, 1, 1),
            nn.Conv2d(512, 30, 1, 1)
        )

        self.layer13_upsample = nn.Sequential(
            ConvBlock(256, 128, 1, 1),
            Upsample(128)
        )

        self.layer26 = nn.Sequential(
            ConvBlock(384, 256, 3, 1, 1),
            nn.Conv2d(256, 30, 1, 1)
        )

    def forward(self, x):
        device = x.device
        backbone26_out = self.backbone26(x)
        backbone13_out = self.backbone13(backbone26_out)
        upsample13 = self.layer13_upsample(backbone13_out)
        stack26 = torch.cat([upsample13, backbone26_out], dim=1).to(device)

        layer13_out = self.layer13(backbone13_out)
        layer26_out = self.layer26(stack26)
        return layer13_out, layer26_out
