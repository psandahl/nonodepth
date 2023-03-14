from __future__ import annotations

import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    """
    Double convolution block, changeing the number of channels.
    """

    def __init__(self: ConvolutionBlock, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding='same',
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding='same',
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self: ConvolutionBlock, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder block. Convolution and downsampling.
    """

    def __init__(self: EncoderBlock, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = ConvolutionBlock(
            in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self: EncoderBlock, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        x = self.pool(skip)

        return skip, x


class DecoderBlock(nn.Module):
    """
    Decoder block. Upsampling and convolution. Takes skip connections from
    the encoder path.
    """

    def __init__(self: DecoderBlock, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = ConvolutionBlock(in_channels=in_channels,
                                     out_channels=out_channels)

    def forward(self: DecoderBlock, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat((x, skip), dim=1)

        return self.conv(x)


class BasicUNet(nn.Module):
    """
    A basic U-Net. The prediction has a tensor with one channel,
    but the same height and with as for the input.
    """

    def __init__(self: BasicUNet, in_channels: int) -> None:
        super().__init__()

        filters = [16, 32, 64, 128, 256]

        self.encoders = nn.ModuleList()
        num_in = in_channels
        for num_out in filters[:-1]:
            self.encoders.append(EncoderBlock(num_in, num_out))
            num_in = num_out

        self.bottleneck = ConvolutionBlock(filters[-2], filters[-1])

        self.decoders = nn.ModuleList()
        num_in = filters[-1]
        for num_out in reversed(filters[:-1]):
            self.decoders.append(DecoderBlock(num_in, num_out))
            num_in = num_out

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[0], out_channels=1, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self: BasicUNet, x: torch.Tensor) -> torch.Tensor:
        skip_connections = list()
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        return self.final_conv(x)
