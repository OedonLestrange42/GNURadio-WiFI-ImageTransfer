import torch
import torch.nn as nn
import torch.nn.functional as F


class AFB(nn.Module):
    def __init__(self, channel_size=128, device=torch.device('cpu')):
        super(AFB, self).__init__()
        self.fc1 = nn.Linear(channel_size + 1, channel_size).to(device)
        self.fc2 = nn.Linear(channel_size, channel_size).to(device)

    def forward(self, SNR, feature):
        x = torch.mean(feature, dim=(2, 3))
        # SNR = torch.tensor(SNR).to(torch.float32).to(device)
        snr = SNR.expand(feature.shape[0], 1)
        x = torch.cat([x, snr], dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        x = x[:, :, None, None]
        x = x.expand(feature.shape)

        return torch.multiply(feature, x)


class AFB_csi(nn.Module):
    def __init__(self, append_size=1, channel_size=128, device=torch.device('cpu')):
        super(AFB_csi, self).__init__()
        tmp = 0
        if not append_size == 0:
            tmp = channel_size
            self.complex2float = nn.Linear(append_size * 2, append_size * 2).to(device)
            self.IRS_compress = nn.Linear(append_size * 2, tmp).to(device)
        self.fc1 = nn.Linear(channel_size + tmp, channel_size).to(device)
        self.fc2 = nn.Linear(channel_size, channel_size).to(device)

    def forward(self, feature, sub_CSI=None):
        x = torch.mean(feature, dim=(2, 3))

        if sub_CSI is not None:
            sub_CSI = torch.cat((sub_CSI.real, sub_CSI.imag), dim=0)
            sub_CSI = self.complex2float(sub_CSI)
            CSI_attention = self.IRS_compress(sub_CSI)
            CSI_attention = CSI_attention.expand(feature.shape[0], CSI_attention.shape[0])
            x = torch.cat([x, CSI_attention], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)

        x = x[:, :, None, None]
        x = x.expand(feature.shape)

        return torch.multiply(feature, x)


class AFB_PE(nn.Module):
    def __init__(self, channel_size=128, device=torch.device('cpu')):
        super(AFB_PE, self).__init__()
        self.fc1 = nn.Linear(channel_size, channel_size).to(device)
        self.fc2 = nn.Linear(channel_size, channel_size).to(device)

    def forward(self, feature, sub_CSI=None):
        x = torch.mean(feature, dim=(2, 3))

        if sub_CSI is not None:
            csi_pe = sub_CSI[:x.shape[1]]
            csi_pe = csi_pe.unsqueeze(0).repeat((x.shape[0], 1))
            x = x + csi_pe

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)

        x = x[:, :, None, None]
        x = x.expand(feature.shape)

        return torch.multiply(feature, x)