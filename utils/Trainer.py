import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn.functional as F
import pdb

from tqdm import tqdm
from math import *
from utils.channel import clustered_SV_channel

random.seed(0)
np.random.seed(0)


def denormalize(img):
    """

    :param img: tensor [b, c, w, h] or [c, w, h]
    :return: denormalized tensor
    """
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    if len(img.shape) == 4:
        img[:, 0, :, :] = std[0] * img[:, 0, :, :] + mean[0]
        img[:, 1, :, :] = std[1] * img[:, 1, :, :] + mean[1]
        img[:, 2, :, :] = std[2] * img[:, 2, :, :] + mean[2]
    elif len(img.shape) == 3:
        img[0] = std[0] * img[0] + mean[0]
        img[1] = std[1] * img[1] + mean[1]
        img[2] = std[2] * img[2] + mean[2]
    else:
        Exception("input must have 3 or 4 channel")

    return img


def psnr(img1, img2):
    epsilon = 1e-4
    psnr = 0
    bs = img1.shape[0]
    if not bs == img2.shape[0]:
        Exception("img1 and img2 have different batch size")
    else:
        for b in range(bs):
            mse = np.mean((img1[b]-img2[b])**2)
            mse = np.max([epsilon, mse])
            psnr += 10.0 * np.log10(255.0 * 255.0 / mse)
    return psnr / bs


def gaussian_policy_loss(log_samples, reward):
    return -(log_samples * reward).mean()


class Trainer():
    def __init__(self,
                 userNum,
                 antennaNum,
                 IRS_scale,
                 SNR,
                 AP_pos,
                 IRS_pos,
                 timesteps=1000,
                 dynamic_userNum=False,
                 dynamic_position=False):
        super(Trainer, self).__init__()
        # env params
        self.interval = 0.03
        self.IRS_scale = IRS_scale
        self.userNum = userNum
        self.atNum = antennaNum
        self.AP_pos = AP_pos
        self.IRS_pos = IRS_pos
        self.Usr_pos = None
        self.Psi = None
        self.H = None
        self.H_U2B_LoS = None
        self.H_R2B_LoS = None
        self.H_U2R_LoS = None
        self.chnl = clustered_SV_channel(self.interval,
                                         IRS_scale,
                                         IRS_pos,
                                         AP_pos,
                                         1,
                                         userNum,
                                         antennaNum)
        self.SNR = SNR

        # dynamic env
        self.dynamic_usr_num = dynamic_userNum
        self.dynamic_usr_pos = dynamic_position

        # params for config training process
        self.max_step = timesteps

    def reset(self, K=10):
        if self.dynamic_usr_num:
            self.userNum = np.random.randint(low=1, high=5)
            self.chnl = clustered_SV_channel(self.interval,
                                             self.IRS_scale,
                                             self.IRS_pos,
                                             self.AP_pos,
                                             1,
                                             self.userNum,
                                             self.atNum)

        if self.dynamic_usr_pos:
            tmp = []
            for i in range(self.userNum):
                tmp.append([np.random.rand()*10, np.random.rand()*10, 1.5])
            self.Usr_pos = np.array(tmp)

            self.H_U2B_LoS, self.H_R2B_LoS, self.H_U2R_LoS = self.chnl.genLoS(self.Usr_pos)
            self.SNR = list(np.random.randint(low=0, high=20, size=(self.userNum,)))

        return self.chnl.genChnl(K), self.SNR

    def train(self, model, image, mode='default', freq_weight=None):
        if mode == 'm2m':
            if hasattr(model, 'mae'):
                _, masked_patch_loss = model(image)
                mean_loss = masked_patch_loss / len(image)
            else:
                output = model(image)
                mean_loss = 0
                for _, s in enumerate(image):
                    mean_loss += F.mse_loss(output[s], image[s])
                mean_loss = mean_loss / len(image)

        elif mode == 'multi-band':
            output = model(image)
            mean_loss = 0
            scaler = 0
            for _, f in enumerate(image):
                mean_loss_freq = 0
                f_weight = freq_weight[f] if freq_weight is not None else 1
                scaler += f_weight
                for _, s in enumerate(image[f]):
                    # TODO: add freq_weight
                    mean_loss_freq += F.mse_loss(output[f][s], image[f][s])
                mean_loss += f_weight * mean_loss_freq / len(image[f])
            mean_loss = mean_loss / scaler
        elif hasattr(model, 'semantic_model'):
            output = model(image)
            mean_loss = 0
            for _, s in enumerate(image):
                mean_loss += model.semantic_model.loss_function(*(output[s]), M_N=0.00025)['loss']
            mean_loss /= len(image)
        else:
            output = model(image)
            mean_loss = F.mse_loss(output, image) / self.userNum

        return mean_loss

    def distill(self, teacher, student, image, decay=0.1):
        output = student(image)
        teacher.eval()
        teacher_output = teacher(image)
        mean_loss = 0
        for _, s in enumerate(image):
            mean_loss += F.mse_loss(output[s], image[s])
            mean_loss += decay * F.mse_loss(output[s], teacher_output[s])
        mean_loss = mean_loss / len(image)

        return mean_loss

    def get_pos(self, withIRS=True, withAP=True, device=torch.device("cpu")):

        if withIRS and withAP:
            Graph = np.vstack((self.AP_pos, self.IRS_pos, self.Usr_pos))
            arr = np.arange(self.userNum + 2)
            x, y = np.meshgrid(arr, arr)
        elif withIRS:
            Graph = np.vstack((self.IRS_pos, self.Usr_pos))
            arr = np.arange(self.userNum + 1)
            x, y = np.meshgrid(arr, arr)
        elif withAP:
            Graph = np.vstack((self.AP_pos, self.Usr_pos))
            arr = np.arange(self.userNum + 1)
            x, y = np.meshgrid(arr, arr)
        else:
            Graph = self.Usr_pos
            arr = np.arange(self.userNum)
            x, y = np.meshgrid(arr, arr)

        edge_index = np.row_stack((x.flatten(), y.flatten()))

        Graph = torch.tensor(Graph, dtype=torch.float32).to(device)
        edge_index = torch.tensor(edge_index, dtype=torch.int64).to(device)

        return Graph, edge_index


if __name__ == "__main__":
    pass
