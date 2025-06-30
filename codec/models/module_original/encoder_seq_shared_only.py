import sys
sys.path.append(r'..')
import torch
import torch.nn as nn
from codec.models.basic_blocks.RCB import RCB
from codec.models.basic_blocks.AFB import AFB_csi
from codec.models.basic_blocks.AFB import AFB_PE


class Encoder(nn.Module):
    def __init__(self, CSI_shape=1, compressed_channel=128, device=torch.device('cpu')):
        super(Encoder, self).__init__()
        self.convBlock1 = RCB(3, 128, 2, device)
        self.attention1 = AFB_csi(CSI_shape, 128, device)
        self.convBlock2 = RCB(128, 256, 2, device)
        self.attention2 = AFB_csi(CSI_shape, 256, device)
        self.convBlock3 = RCB(256, 256, 2, device)
        self.attention3 = AFB_csi(CSI_shape, 256, device)
        self.convBlock4 = RCB(256, compressed_channel, 1, device)

    def forward(self, r, sub_CSI=None):
        s = self.convBlock1(r)
        s = self.attention1(s, sub_CSI)
        s = self.convBlock2(s)
        s = self.attention2(s, sub_CSI)
        s = self.convBlock3(s)
        s = self.attention3(s, sub_CSI)
        s = self.convBlock4(s)
        return s

class Encoder_PE(nn.Module):
    def __init__(self, compressed_channel=128, input_channel=3, device=torch.device('cpu')):
        super(Encoder_PE, self).__init__()
        self.convBlock1 = RCB(input_channel, 128, 2, device)
        self.attention1 = AFB_PE(128, device)
        self.convBlock2 = RCB(128, 256, 2, device)
        self.attention2 = AFB_PE(256, device)
        self.convBlock3 = RCB(256, 256, 2, device)
        self.attention3 = AFB_PE(256, device)
        self.convBlock4 = RCB(256, compressed_channel, 1, device)

    def forward(self, r, sub_CSI=None):
        s = self.convBlock1(r)
        s = self.attention1(s, sub_CSI)
        s = self.convBlock2(s)
        s = self.attention2(s, sub_CSI)
        s = self.convBlock3(s)
        s = self.attention3(s, sub_CSI)
        s = self.convBlock4(s)
        return s
