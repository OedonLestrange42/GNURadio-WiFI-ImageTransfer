import sys
sys.path.append(r'..')
import torch
import torch.nn as nn
from basic_blocks.RTCB import RTCB
from basic_blocks.AFB import AFB


class Decoder(nn.Module):
    def __init__(self, SNR, compressed_channel=128, reconstruct_channel=3, device=torch.device('cpu')):
        super(Decoder, self).__init__()
        self.proc = nn.Sequential(
            RTCB(compressed_channel, 256, 1, device),
            AFB(SNR, 256, device),
            RTCB(256, 256,2, device),
            AFB(SNR, 256, device),
            RTCB(256, 128, 2, device),
            AFB(SNR, 128, device),
            RTCB(128, reconstruct_channel, 2, device)
        )

    def to(self, device):
        self.proc.to(device)
        return self

    def forward(self, s):
        r = self.proc(s)
        return r

        