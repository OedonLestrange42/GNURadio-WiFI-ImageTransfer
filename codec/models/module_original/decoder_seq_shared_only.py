import sys
sys.path.append(r'..')
import torch
import torch.nn as nn
from codec.models.basic_blocks.RTCB import RTCB
from codec.models.basic_blocks.AFB import AFB_csi
from codec.models.basic_blocks.AFB import AFB_PE


class Decoder(nn.Module):
    def __init__(self, CSI_shape=1, compressed_channel=128, reconstruct_channel=3, device=torch.device('cpu')):
        super(Decoder, self).__init__()
        self.convBlock1 = RTCB(compressed_channel, 256, 1, device)
        self.attention1 = AFB_csi(CSI_shape, 256, device)
        self.convBlock2 = RTCB(256, 256, 2, device)
        self.attention2 = AFB_csi(CSI_shape, 256, device)
        self.convBlock3 = RTCB(256, 128, 2, device)
        self.attention3 = AFB_csi(CSI_shape, 128, device)
        self.convBlock4 = RTCB(128, reconstruct_channel, 2, device)

    def forward(self, s, sub_CSI):
        r = self.convBlock1(s)
        # r = torch.clamp(r, min=-1e-4, max=1e+4)
        r = self.attention1(r, sub_CSI)
        r = self.convBlock2(r)
        r = self.attention2(r, sub_CSI)
        r = self.convBlock3(r)
        r = self.attention3(r, sub_CSI)
        r = self.convBlock4(r)
        return r


class Decoder_PE(nn.Module):
    def __init__(self, compressed_channel=128, reconstruct_channel=3, device=torch.device('cpu')):
        super(Decoder_PE, self).__init__()
        self.convBlock1 = RTCB(compressed_channel, 256, 1, device)
        self.attention1 = AFB_PE(256, device)
        self.convBlock2 = RTCB(256, 256, 2, device)
        self.attention2 = AFB_PE(256, device)
        self.convBlock3 = RTCB(256, 128, 2, device)
        self.attention3 = AFB_PE(128, device)
        self.convBlock4 = RTCB(128, reconstruct_channel, 2, device)

    def forward(self, s, sub_CSI):
        r = self.convBlock1(s)
        # r = torch.clamp(r, min=-1e-4, max=1e+4)
        r = self.attention1(r, sub_CSI)
        r = self.convBlock2(r)
        r = self.attention2(r, sub_CSI)
        r = self.convBlock3(r)
        r = self.attention3(r, sub_CSI)
        r = self.convBlock4(r)
        return r

class Decoder_SA(nn.Module):
    def __init__(self, compressed_channel=128, reconstruct_channel=3, device=torch.device('cpu')):
        super(Decoder_SA, self).__init__()
        self.convBlock1 = RTCB(compressed_channel, 256, 1, device)
        self.convBlock2 = RTCB(256, 256, 2, device)
        self.convBlock3 = RTCB(256, 128, 2, device)
        self.convBlock4 = RTCB(128, reconstruct_channel, 2, device)

    def forward(self, s):
        r = self.convBlock1(s)
        r = self.convBlock2(r)
        r = self.convBlock3(r)
        r = self.convBlock4(r)
        return r