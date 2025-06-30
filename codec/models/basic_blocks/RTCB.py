import torch
import torch.nn as nn
from .GDN import GDN


class RTCB(nn.Module):
    """
    based on Resnet BasicBlock
    """
    expansion: int = 1

    def __init__(self, in_channel=3, out_channel=128, stride=2, device=torch.device('cpu')):
        super(RTCB, self).__init__()
        self.proc = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, output_padding=stride-1).to(device),
            GDN(out_channel, inverse=True, device=device),
            nn.PReLU(num_parameters=out_channel, device=device),
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1).to(device),
            GDN(out_channel, inverse=True, device=device),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel * self.expansion:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 1, stride=stride, padding=0, output_padding=stride-1).to(device),
                GDN(out_channel, inverse=True, device=device)
            )

        self.fn = nn.PReLU(num_parameters=out_channel, device=device)
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        y = self.proc(x)
        z = self.shortcut(x)
        return self.fn(y + z)

