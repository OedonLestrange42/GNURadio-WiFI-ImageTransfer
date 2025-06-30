import torch.nn as nn
import torch


class PropagNet(nn.Module):
    def __init__(self, env, withIRS=True):
        super(PropagNet, self).__init__()

        self.H_B2R = torch.complex(torch.FloatTensor(env[0].real), torch.FloatTensor(env[0].imag))
        self.H_R2U = torch.complex(torch.FloatTensor(env[1].real), torch.FloatTensor(env[1].imag))
        self.Hd = torch.complex(torch.FloatTensor(env[2].real), torch.FloatTensor(env[2].imag))
        self.withIRS = withIRS
        self.env = env
        if withIRS:
            self.phi = nn.Parameter(2 * 3.14 * torch.rand(env[0].shape[1]), requires_grad=True)
            self.psi_real = torch.cos(self.phi)
            self.psi_imag = torch.sin(self.phi)
        else:
            self.psi_real = torch.zeros(env[0].shape[1])
            self.psi_imag = torch.zeros(env[0].shape[1])

    def to(self, device):
        # 调用父类方法将模型参数移动到指定设备
        super(PropagNet, self).to(device)
        # 手动将常数变量移动到指定设备
        self.H_B2R = self.H_B2R.to(device)
        self.H_R2U = self.H_R2U.to(device)
        self.Hd = self.Hd.to(device)
        if not self.withIRS:
            self.psi_real = self.psi_real.to(device)
            self.psi_imag = self.psi_imag.to(device)
        else:
            self.phi = self.phi.to(device)
        return self

    def get_CSI(self, index=0, Psi=None):
        if self.withIRS:
            self.psi_real = torch.cos(self.phi)
            self.psi_imag = torch.sin(self.phi)
        if Psi is None:
            Psi = torch.complex(self.psi_real, self.psi_imag)
        H = torch.linalg.multi_dot([self.H_B2R, torch.diag(Psi), self.H_R2U[:, index]]) + self.Hd[:, index]
        return H

    def refresh_Psi(self, seed):
        torch.manual_seed(seed)
        self.phi = nn.Parameter(2 * 3.14 * torch.rand(self.env[0].shape[1]), requires_grad=False)
        return self.phi.data

    def reset(self, env):
        self.H_B2R = torch.complex(torch.FloatTensor(env[0].real), torch.FloatTensor(env[0].imag))
        self.H_R2U = torch.complex(torch.FloatTensor(env[1].real), torch.FloatTensor(env[1].imag))
        self.Hd = torch.complex(torch.FloatTensor(env[2].real), torch.FloatTensor(env[2].imag))
        self.env = env

    def forward(self, x):
        CSI = []
        if self.withIRS:
            self.psi_real = torch.cos(self.phi)
            self.psi_imag = torch.sin(self.phi)
        Psi = torch.complex(self.psi_real, self.psi_imag)

        # path n
        path_num = self.Hd.shape[1]
        Y = []
        for p in range(path_num):
            H = torch.linalg.multi_dot([self.H_B2R, torch.diag(Psi), self.H_R2U[:, p]]) + self.Hd[:, p]
            y = H * x

            Y.append(y)
            CSI.append(H)

        return Y, CSI
