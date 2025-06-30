import torch.nn as nn

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.ConvTranspose2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class RTCB(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act=nn.Hardswish, se=False, stride=1):
        super(RTCB, self).__init__()
        self.stride = stride

        self.conv1 = nn.ConvTranspose2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.ConvTranspose2d(expand_size, expand_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=1,
                                        output_padding=stride - 1,
                                        groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.ConvTranspose2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_size,
                                   out_channels=in_size,
                                   kernel_size=3,
                                   groups=in_size,
                                   stride=2,
                                   padding=1,
                                   output_padding=stride - 1,
                                   bias=False),
                nn.BatchNorm2d(in_size),
                nn.ConvTranspose2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_size,
                                   out_channels=out_size,
                                   kernel_size=3,
                                   groups=in_size,
                                   stride=2,
                                   padding=1,
                                   output_padding=stride - 1,
                                   bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


if __name__ == '__main__':
    import torch
    img = torch.randn(1, 128, 32, 32)

    net1 = RTCB(kernel_size=3,
                 in_size=128,
                 expand_size=256,
                 out_size=128)

    net2 = RTCB(kernel_size=3,
                 in_size=128,
                 expand_size=256,
                 out_size=3,
                 stride=2)

    out = net1(img)
