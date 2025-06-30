import sys
sys.path.append(r'..')
import torch
import torch.nn as nn
from semantic_model.basic_blocks.RTCB_DS import RTCB
from semantic_model.basic_blocks.AFB import AFB_PE


class Decoder(nn.Module):
    def __init__(self, compressed_channel=128, reconstruct_channel=3, device=torch.device('cpu')):
        super(Decoder, self).__init__()
        self.convBlock1 = RTCB(kernel_size=3,
                               in_size=compressed_channel,
                               expand_size=256,
                               out_size=256,
                               stride=2).to(device)
        self.attention1 = AFB_PE(256, device)
        self.convBlock2 = RTCB(kernel_size=3,
                               in_size=256,
                               expand_size=512,
                               out_size=256,
                               stride=2).to(device)
        self.attention2 = AFB_PE(256, device)
        self.convBlock3 = RTCB(kernel_size=3,
                               in_size=256,
                               expand_size=256,
                               out_size=128,
                               stride=2).to(device)
        self.attention3 = AFB_PE(128, device)
        self.convBlock4 = RTCB(kernel_size=3,
                               in_size=128,
                               expand_size=96,
                               out_size=reconstruct_channel,
                               stride=1).to(device)

    def forward(self, s, sub_CSI=None):
        r = self.convBlock1(s)
        # r = torch.clamp(r, min=-1e-4, max=1e+4)
        r = self.attention1(r, sub_CSI)
        r = self.convBlock2(r)
        r = self.attention2(r, sub_CSI)
        r = self.convBlock3(r)
        r = self.attention3(r, sub_CSI)
        r = self.convBlock4(r)
        return r


if __name__ == '__main__':
    import torch
    img = torch.randn(1, 128, 32, 32)

    net1 = Decoder()
    out1 = net1(img)