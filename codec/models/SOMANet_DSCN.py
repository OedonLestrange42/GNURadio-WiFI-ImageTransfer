import torch
import torch.nn as nn
import math
from .module_original.encoder_dscn import Encoder
from .module_original.decoder_seq_shared_only import Decoder_PE as Decoder


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class DMANet(nn.Module):
    def __init__(self, envs, img_size=256, compressed_channel=128, P=1, CSI_bound=30, withIRS=False, optimizeIRS=True, device=torch.device('cpu')):
        super(DMANet, self).__init__()
        self.compressed_channel = compressed_channel
        self.device = device
        self.withIRS = withIRS
        self.P = P
        self.envs = [torch.tensor(h, dtype=torch.complex64).to(device) for h in envs]
        self.img_size = img_size

        self.shared_encoder = Encoder(compressed_channel=compressed_channel,
                                      device=self.device)
        self.shared_decoder = Decoder(compressed_channel=compressed_channel,
                                      reconstruct_channel=3,
                                      device=self.device)

        self.CSI_bound = CSI_bound
        self.PE = positionalencoding2d(d_model=256, height=2*self.CSI_bound, width=2*self.CSI_bound).to(device)

        # envs: list [g1, g2, ..., gK]
        if optimizeIRS == True:
            self.shared_phi = nn.Parameter(2 * 3.14 * torch.rand(envs[0].shape[0]).to(device), requires_grad=True)
        elif optimizeIRS == False:
            self.shared_phi = nn.Parameter(2 * 3.14 * torch.rand(envs[0].shape[0]).to(device), requires_grad=False)

    def to(self, device):
        self.device = device
        self.H = self.H.to(device)
        self.shared_decoder.to(self.device)
        self.shared_encoder.to(self.device)
        return self

    def refresh_rician_channel(self, envs):
        self.envs = [torch.tensor(h, dtype=torch.complex64).to(self.device) for h in envs]

    def powerNorm(self, feature):
        mod = torch.pow(torch.abs(feature),2).sum(dim=1, keepdim=True).expand_as(feature).to(torch.float32)
        epsilon = torch.tensor(1e-6).to(torch.float32).to(self.device)
        mod = torch.where(mod == 0, epsilon, mod)
        weight = torch.tensor(self.P * feature.shape[1])
        feature = torch.sqrt(weight) * feature / mod
        return feature

    def noise(self, SNR, shape):
        sigma = (10 ** (-1 * SNR / 10) * self.P)
        n = 1 / torch.sqrt(torch.tensor(2)) * (torch.normal(0, sigma, size=shape) + 1j * torch.normal(0, sigma, size=shape))
        return n.to(self.device)

    def getChnl(self, tx, rx):
        Psi = torch.complex(torch.cos(self.shared_phi), torch.sin(self.shared_phi))
        CSI = torch.linalg.multi_dot([torch.transpose(self.envs[int(rx)], 0, 1),
                                      torch.diag(Psi),
                                      self.envs[int(tx)]])
        return CSI

    def getCSI(self, key):
        real, imag = key.split('-')
        return torch.complex(torch.tensor([float(real)]), torch.tensor([float(imag)]))

    def forward(self, schedule_list):

        """

        :param schedule_list: schedule dict {'i-j': p_ij} where gi, gj are the user channel, p_ij is the image
        :return: reconstruction_result: a dict like {'i-j': p'_ij} where p'_ij is the reconstruction results
        """
        user_result = {}

        Psi = torch.complex(torch.cos(self.shared_phi), torch.sin(self.shared_phi))

        # encode
        featurelist = []
        for i, k in enumerate(schedule_list):
            CSI = self.getCSI(k)
            w = torch.clamp(torch.round(CSI.real[0]).detach() + self.CSI_bound, 0, 2*self.CSI_bound-1).to(torch.int)
            h = torch.clamp(torch.round(CSI.imag[0]).detach() + self.CSI_bound, 0, 2*self.CSI_bound-1).to(torch.int)
            label = self.PE[:, w, h].squeeze()
            feature = self.shared_encoder(schedule_list[k], label)

            s = feature.view(feature.shape[0], -1)  # flatten
            l = s.shape[1] // 2
            s = torch.complex(s[:, :l], s[:, l:])
            featurelist.append(s)

        # merger & propagation
        MSSV = torch.stack(featurelist)
        MSSV = torch.sum(MSSV, dim=0)
        MSSV = self.powerNorm(MSSV)

        """# reflecting & receiving
        psi = torch.diag_embed(Psi)
        psi = Psi.expand((MSSV.shape[0], psi.shape[0], psi.shape[1]))
        reflect_sig = torch.bmm(psi, MSSV)"""

        for i, k in enumerate(schedule_list):
            # CSI
            CSI = self.getCSI(k)
            w = torch.clamp(torch.round(CSI.real[0]).detach() + self.CSI_bound, 0, 2 * self.CSI_bound - 1).to(torch.int)
            h = torch.clamp(torch.round(CSI.imag[0]).detach() + self.CSI_bound, 0, 2 * self.CSI_bound - 1).to(torch.int)
            label = self.PE[:, w, h].squeeze()

            # reshape
            recv = (torch.cat((MSSV.real, MSSV.imag), dim=1)
                    .view(MSSV.size(0), self.compressed_channel, self.img_size // 8, self.img_size // 8))

            # decode
            x = nn.functional.normalize(recv, p=2, dim=1)
            x = self.shared_decoder(x, label)
            user_result[k] = x

        return user_result


    def _forward_old(self, schedule_list):

        """

        :param schedule_list: schedule dict {'i-j': p_ij} where gi, gj are the user channel, p_ij is the image
        :return: reconstruction_result: a dict like {'i-j': p'_ij} where p'_ij is the reconstruction results
        """
        user_result = {}

        Psi = torch.complex(torch.cos(self.shared_phi), torch.sin(self.shared_phi))

        signal_list = []
        channel_list = []
        # encode
        for i, k in enumerate(schedule_list):
            CSI = torch.linalg.multi_dot([torch.transpose(self.envs[int(k.split('-')[0])], 0, 1),
                                          torch.diag(Psi),
                                          self.envs[int(k.split('-')[1])]])
            w = torch.clamp(torch.round(CSI.real[0]).detach() + self.CSI_bound, 0, 2*self.CSI_bound-1).to(torch.int)
            h = torch.clamp(torch.round(CSI.imag[0]).detach() + self.CSI_bound, 0, 2*self.CSI_bound-1).to(torch.int)
            label = self.PE[:, w, h].squeeze()
            feature = self.shared_encoder(schedule_list[k], label)

            s = feature.view(feature.shape[0], -1)  # flatten
            l = s.shape[1] // 2
            s = torch.complex(s[:, :l], s[:, l:])
            sk = self.powerNorm(s).unsqueeze(1)

            g = self.envs[int(k.split('-')[0])]
            g = g.expand((sk.shape[0], g.shape[0], g.shape[1]))

            signal_list.append(sk)
            channel_list.append(g)

        # merger & propagation
        sig = torch.cat(signal_list, dim=1)
        H = torch.cat(channel_list, dim=2)
        MSSV = torch.bmm(H, sig)

        # reflecting & receiving
        psi = torch.diag_embed(Psi)
        psi = Psi.expand((MSSV.shape[0], psi.shape[0], psi.shape[1]))
        reflect_sig = torch.bmm(psi, MSSV)

        for i, k in enumerate(schedule_list):
            # CSI
            CSI = torch.linalg.multi_dot([torch.transpose(self.envs[int(k.split('-')[0])], 0, 1),
                                          torch.diag(Psi),
                                          self.envs[int(k.split('-')[1])]])
            w = torch.clamp(torch.round(CSI.real[0]).detach() + self.CSI_bound, 0, 2 * self.CSI_bound - 1).to(torch.int)
            h = torch.clamp(torch.round(CSI.imag[0]).detach() + self.CSI_bound, 0, 2 * self.CSI_bound - 1).to(torch.int)
            label = self.PE[:, w, h].squeeze()

            g = torch.transpose(self.envs[int(k.split('-')[1])], 0, 1)
            g = g.expand((MSSV.shape[0], g.shape[0], g.shape[1]))
            recv_sig = torch.bmm(g, reflect_sig)
            """SINR = self.get_SINR(k, schedule_list)  # TODO: get SINR"""
            recv_sig += self.noise(20, recv_sig.shape)
            recv_sig = torch.matmul(torch.inverse(CSI), recv_sig)
            # reshape
            recv = (torch.cat((recv_sig.real, recv_sig.imag), dim=1)
                    .view(recv_sig.size(0), self.compressed_channel, self.img_size // 8, self.img_size // 8))

            # decode
            x = nn.functional.normalize(recv, p=2, dim=1)
            x = self.shared_decoder(x, label)
            user_result[k] = x

        return user_result

