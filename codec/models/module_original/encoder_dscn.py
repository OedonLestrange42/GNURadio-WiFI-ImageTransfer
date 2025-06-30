import sys
sys.path.append(r'..')
import torch
import torch.nn as nn
from codec.models.basic_blocks.RCB_DS import RCB
from codec.models.basic_blocks.AFB import AFB_PE


class Encoder(nn.Module):
    def __init__(self, compressed_channel=128, input_channel=3, device=torch.device('cpu')):
        super(Encoder, self).__init__()
        self.convBlock1 = RCB(kernel_size=3,
                              in_size=input_channel,
                              expand_size=96,
                              out_size=128,
                              stride=2).to(device)
        self.attention1 = AFB_PE(128, device)
        self.convBlock2 = RCB(kernel_size=3,
                              in_size=128,
                              expand_size=256,
                              out_size=256,
                              stride=2).to(device)
        self.attention2 = AFB_PE(256, device)
        self.convBlock3 = RCB(kernel_size=3,
                              in_size=256,
                              expand_size=512,
                              out_size=256,
                              stride=2).to(device)
        self.attention3 = AFB_PE(256, device)
        self.convBlock4 = RCB(kernel_size=3,
                              in_size=256,
                              expand_size=256,
                              out_size=compressed_channel,
                              stride=1).to(device)

    def forward(self, r, sub_CSI=None):
        s = self.convBlock1(r)
        s = self.attention1(s, sub_CSI)
        s = self.convBlock2(s)
        s = self.attention2(s, sub_CSI)
        s = self.convBlock3(s)
        s = self.attention3(s, sub_CSI)
        s = self.convBlock4(s)
        return s


if __name__ == '__main__':
    import torch
    img = torch.randn(1, 3, 256, 256)

    net1 = Encoder()
    out1 = net1(img)