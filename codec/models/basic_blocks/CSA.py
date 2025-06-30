# 'CSA' stands for Channel-Spatial Attention

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFB(nn.Module):
    def __init__(self, SNR, channel_size=128, device=torch.device('cpu')):
        super(AFB, self).__init__()
        self.multiSNR = False
        if isinstance(SNR, list):
            self.multiSNR = True
            self.fc1 = nn.Linear(channel_size + len(SNR), channel_size).to(device)
        else:
            self.fc1 = nn.Linear(channel_size + 1, channel_size).to(device)
        self.fc2 = nn.Linear(channel_size, channel_size).to(device)
        self.SNR = torch.tensor(SNR).to(torch.float32).to(device)

    def forward(self, feature):
        x = torch.mean(feature, dim=(2, 3))
        if self.multiSNR:
            snr = self.SNR.expand(feature.shape[0], self.SNR.shape[0])
        else:
            snr = self.SNR.expand(feature.shape[0], 1)
        x = torch.cat([x, snr], dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        x = x[:, :, None, None]
        x = x.expand(feature.shape)

        return torch.multiply(feature, x)