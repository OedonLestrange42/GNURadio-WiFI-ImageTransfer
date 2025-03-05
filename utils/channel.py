import numpy as np
import random
import datetime
import pdb
from math import *

random.seed(0)
np.random.seed(0)


def genDFT(num):
    DFT = np.exp(1j * pi * np.zeros((num, num)))
    for i in range(num):
        for j in range(num):
            DFT[i, j] = np.exp(-1j * 2 * pi * (i + 1) * (j + 1) / (num + 1))
    return DFT


def genBImatrix(num):
    """
    生成0~2^num的所有数字的二进制构成的二维矩阵
    :param num: 位数，当num<8时可以直接生成，>8时需要拼接
    :return: 0~2^num的所有数字的二进制构成的二维矩阵
    """
    if num < 8:
        tmp1 = np.unpackbits(np.arange(2 ** num).astype(np.uint8)[:, None], axis=1)[:, -1 * num:]
    else:
        tmp1 = np.unpackbits(np.arange(2 ** 8).astype(np.uint8)[:, None], axis=1)
        while num > 8:
            num -= 8
            tmp2 = [[np.hstack((i, j)) for j in tmp1] for i in tmp1]
            tmp2 = [np.vstack(x) for x in tmp2]
            tmp1 = np.vstack(tmp2)
    return tmp1


def get_moving_average(mylist, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(mylist, window, 'valid')
    return re


class Channel():
    def __init__(self, userNum, antennaNum, IRS_units):
        self.N = IRS_units
        self.scale = int(self.N ** 0.5)
        self.userNum = userNum
        self.atNum = antennaNum

    def noise(self, sigma):
        n = 1 / sqrt(2) * (np.random.normal(0, sigma, size=(self.atNum, self.userNum))
                           + 1j * np.random.normal(0, sigma, size=( self.atNum,self.userNum)))
        return n

    def Steervec(self, angle, atNum):
        sv = np.exp(1j * angle * pi * np.arange(0, atNum, 1))
        sv = sv.reshape(-1, 1)
        return sv

    def ChannelMdl(self, pos_A, pos_B, atScale_A, atScale_B, f=5e9):
        c = 3e8
        k = 2 * np.pi * f / c
        dis_AB = np.linalg.norm(pos_A - pos_B)  # distance
        n_AB = (pos_A - pos_B) / dis_AB  # direction vector
        angleA = [np.linalg.multi_dot([[1, 0, 0], n_AB]),
                  np.linalg.multi_dot([[0, 1, 0], n_AB]),
                  np.linalg.multi_dot([[0, 0, 1], n_AB])]

        sv_A = np.kron(self.Steervec(angleA[0], atScale_A[0]), self.Steervec(angleA[1], atScale_A[1]))
        sv_A = np.kron(sv_A, self.Steervec(angleA[2], atScale_A[2]))
        angleB = [np.linalg.multi_dot([[1, 0, 0], n_AB]),
                  np.linalg.multi_dot([[0, 1, 0], n_AB]),
                  np.linalg.multi_dot([[0, 0, 1], n_AB])]
        sv_B = np.kron(self.Steervec(angleB[0], atScale_B[0]), self.Steervec(angleB[1], atScale_B[1]))
        sv_B = np.kron(sv_B, self.Steervec(angleB[2], atScale_B[2]))

        H = np.linalg.multi_dot([sv_A, np.matrix.getH(sv_B)]) * np.exp(-1j * k * dis_AB)
        return H

    def propagation(self, signal, H, sigma):
        n = self.noise(sigma)
        y = np.dot(H, signal) + n
        return y

    # Non LoS channel
    def genNonLoS(self):
        H_B2U_NLoS = 1 / sqrt(2) * (
                    np.random.normal(0, 1, size=(self.atNum, self.userNum)) + 1j * np.random.normal(0, 1, size=(
            self.atNum, self.userNum)))
        H_B2R_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(self.atNum, self.N)) + 1j * np.random.normal(0, 1,
                                                                                                              size=(
                                                                                                              self.atNum,
                                                                                                              self.N)))
        H_R2U_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(self.N, self.userNum)) + 1j * np.random.normal(0, 1,
                                                                                                                size=(
                                                                                                                self.N,
                                                                                                                self.userNum)))
        return H_B2U_NLoS, H_B2R_NLoS, H_R2U_NLoS

    # LoS channel
    def genLoS(self, pos_AP, pos_IRS, pos_User):
        atScale_User = [1, 1, 1]
        atScale_AP = [self.atNum, 1, 1]
        atScale_IRS = [self.scale, self.scale, 1]

        H_B2U_LoS = np.zeros((self.atNum, self.userNum)) + 1j * np.zeros((self.atNum, self.userNum))
        H_U2R_LoS = np.zeros((self.N, self.userNum)) + 1j * np.zeros((self.N, self.userNum))
        for iu in range(self.userNum):
            h_B2U_LoS = self.ChannelMdl(pos_AP, pos_User[iu], atScale_AP, atScale_User)
            H_B2U_LoS[:, iu] = h_B2U_LoS.reshape(-1)
            h_U2R_LoS = self.ChannelMdl(pos_IRS, pos_User[iu], atScale_IRS, atScale_User)
            H_U2R_LoS[:, iu] = h_U2R_LoS.reshape(-1)
        H_B2R_LoS = self.ChannelMdl(pos_AP, pos_IRS, atScale_AP, atScale_IRS)

        return H_B2U_LoS, H_B2R_LoS, H_U2R_LoS

    # aggregate channel
    def getChnl(self, H_B2U, H_B2R, H_R2U, psi):
        Psi = np.diag(psi.flatten())
        H = np.linalg.multi_dot([H_B2R, Psi, H_R2U]) + H_B2U
        return H

    def DFT_matrix(self, Node):
        n, m = np.meshgrid(np.arange(Node), np.arange(Node))
        omega = np.exp(-2 * pi * 1j / Node)
        W = np.power(omega, n * m) / sqrt(Node)
        return W

    def CH_est(self, y_rx, sigma2, Pilot):  # 评估
        MMSE_matrix = np.matrix.getH(Pilot) / (1 + sigma2)  # MMSE channel estimation
        H_est = np.dot(y_rx, MMSE_matrix)
        return H_est


class clustered_SV_channel():
    def __init__(self, IRS_unit_gap, IRS_scale, IRS_pos, AP_pos, cluster_cale, usr_num, AP_num):
        self.group_scale = IRS_scale // cluster_cale
        self.group_num = self.group_scale ** 2
        self.group_size = cluster_cale
        self.IRS_scale = IRS_scale
        self.N = IRS_scale ** 2
        self.IRS_shape = [IRS_scale, IRS_scale, 1]
        self.IRS_pos = IRS_pos  # shape: [3, ] representing the coordinate of first element (the left-top one in right-hand coordinate)
        self.AP_pos = AP_pos
        self.interval = IRS_unit_gap
        self.freq = 5e9
        self.Y = np.reshape(np.array([list(range(self.group_scale)) * self.group_scale]),
                            [self.group_scale, self.group_scale])
        self.X = np.transpose(self.Y)
        self.pos = [  # [x0,y0,z0] is the matrix of each IRS unit's position
            (np.ones((self.group_scale, self.group_scale)) * IRS_pos[0] + self.X * self.interval),  # x0
            (np.ones((self.group_scale, self.group_scale)) * IRS_pos[1] + self.Y * self.interval),  # y0
            (np.ones((self.group_scale, self.group_scale)) * IRS_pos[2])]
        self.chnl = Channel(usr_num, AP_num, 1)
        self.atNum = AP_num
        self.userNum = usr_num

        self.H_U2B_LoS = np.zeros((self.atNum, self.userNum), dtype=complex)
        self.H_R2B_LoS = np.zeros((self.atNum, self.group_num), dtype=complex)
        self.H_U2R_LoS = np.zeros((self.group_num, self.userNum), dtype=complex)

    def genLoS(self, pos_User):
        atScale_AP = [1, 1, 1]
        atScale_User = [1, 1, 1]
        atScale_group = [self.group_size, self.group_size, 1]

        self.H_U2B_LoS = np.zeros((self.atNum, self.userNum)) + 1j * np.zeros((self.atNum, self.userNum))
        for iu in range(self.userNum):
            h_B2U_LoS = self.chnl.ChannelMdl(self.AP_pos, pos_User[iu], atScale_AP, atScale_User)[0][0]
            self.H_U2B_LoS[:, iu] = h_B2U_LoS.reshape(-1)

        for i in range(self.group_scale):
            for j in range(self.group_scale):
                pos_unit = [self.pos[0][i, j], self.pos[1][i, j], self.pos[2][i, j]]
                for iu in range(self.userNum):
                    h_U2R_LoS = self.chnl.ChannelMdl(pos_User[iu], pos_unit, atScale_AP, atScale_group)[0][0]
                    self.H_U2R_LoS[i * self.group_scale + j, iu] = h_U2R_LoS
                h_R2B_LoS = self.chnl.ChannelMdl(self.AP_pos, pos_unit, atScale_AP, atScale_group)[0][0]
                self.H_R2B_LoS[:, i * self.group_scale + j] = h_R2B_LoS

        return self.H_U2B_LoS, self.H_R2B_LoS, self.H_U2R_LoS

    def genLoS_tmp(self, pos_User):
        atScale_AP = [1, 1, 1]
        atScale_User = [1, 1, 1]
        atScale_group = [self.IRS_scale, self.IRS_scale, 1]

        self.H_U2B_LoS = np.zeros((self.atNum, self.userNum)) + 1j * np.zeros((self.atNum, self.userNum))

        pos_unit = [self.pos[0][0,0], self.pos[1][0,0], self.pos[2][0,0]]
        for iu in range(self.userNum):
            h_U2R_LoS = self.chnl.ChannelMdl(pos_User[iu], pos_unit, atScale_AP, atScale_group)[0][0]
            self.H_U2R_LoS[:, iu] = h_U2R_LoS
            h_B2U_LoS = self.chnl.ChannelMdl(self.AP_pos, pos_User[iu], atScale_AP, atScale_User)[0][0]
            self.H_U2B_LoS[:, iu] = h_B2U_LoS.reshape(-1)
        h_R2B_LoS = self.chnl.ChannelMdl(self.AP_pos, pos_unit, atScale_AP, atScale_group)
        self.H_R2B_LoS = h_R2B_LoS

        return self.H_U2B_LoS, self.H_R2B_LoS, self.H_U2R_LoS

    def genNonLoS(self):
        H_R2B_NLoS = 1 / sqrt(2) * (
                    np.random.normal(0, 1, size=(self.atNum, self.group_num)) + 1j * np.random.normal(0, 1, size=(
            self.atNum, self.group_num)))
        H_U2R_NLoS = 1 / sqrt(2) * (
                    np.random.normal(0, 1, size=(self.group_num, self.userNum)) + 1j * np.random.normal(0, 1, size=(
            self.group_num, self.userNum)))
        return H_R2B_NLoS, H_U2R_NLoS

    def genChnl(self, K=10):
        H_U2B_NLoS = 1 / sqrt(2) * (
                    np.random.normal(0, 1, size=(self.atNum, self.userNum)) + 1j * np.random.normal(0, 1, size=(
            self.atNum, self.userNum)))
        H_d = sqrt(K / (K + 1)) * self.H_U2B_LoS + sqrt(1 / (K + 1)) * H_U2B_NLoS

        H_R2B_NLoS, H_U2R_NLoS = self.genNonLoS()
        H_R2B = sqrt(K / (K + 1)) * self.H_R2B_LoS + sqrt(1 / (K + 1)) * H_R2B_NLoS
        H_U2R = sqrt(K / (K + 1)) * self.H_U2R_LoS + sqrt(1 / (K + 1)) * H_U2R_NLoS

        return H_R2B, H_U2R, H_d

    def genChnl_cast_only(self, Psi, K=10):
        H_R2B_NLoS, H_U2R_NLoS = self.genNonLoS()
        H_R2B = sqrt(K / (K + 1)) * self.H_R2B_LoS + sqrt(1 / (K + 1)) * H_R2B_NLoS
        H_U2R = sqrt(K / (K + 1)) * self.H_U2R_LoS + sqrt(1 / (K + 1)) * H_U2R_NLoS
        """H_R2B = self.H_R2B_LoS
        H_U2R = self.H_U2R_LoS"""
        block = np.diag(Psi.flatten())
        H_r = np.linalg.multi_dot([H_R2B, block, H_U2R])
        return H_r

    def genChnl_ideal(self, Psi):
        H_R2B = self.H_R2B_LoS
        H_U2R = self.H_U2R_LoS
        block = np.diag(Psi.flatten())
        H_r = np.linalg.multi_dot([H_R2B, block, H_U2R])
        return H_r

    def genSubChnl(self, K=10):
        H_U2B_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(self.atNum, self.userNum)) +
                                    1j * np.random.normal(0, 1, size=(self.atNum, self.userNum)))
        H_R2B_NLoS, H_U2R_NLoS = self.genNonLoS()
        H_R2B = sqrt(K / (K + 1)) * self.H_R2B_LoS + sqrt(1 / (K + 1)) * H_R2B_NLoS
        H_U2R = sqrt(K / (K + 1)) * self.H_U2R_LoS + sqrt(1 / (K + 1)) * H_U2R_NLoS
        H_d = sqrt(K / (K + 1)) * self.H_U2B_LoS + sqrt(1 / (K + 1)) * H_U2B_NLoS
        return H_R2B, H_U2R, H_d

    def genPhase_nograd(self, theta, phi):
        # radians rather than degree!
        k = 2 * np.pi * self.freq / 3e8
        atpos = [  # [x1,y1,z1] is the matrix of AP's position
            (np.ones((self.IRS_scale, self.IRS_scale)) * self.AP_pos[0]),  # x1
            (np.ones((self.IRS_scale, self.IRS_scale)) * self.AP_pos[1]),  # y1
            (np.ones((self.IRS_scale, self.IRS_scale)) * self.AP_pos[2])]  # z1

        D = np.power(np.power((self.pos[0] - atpos[0]), 2) + \
                     np.power((self.pos[1] - atpos[1]), 2) + \
                     np.power((self.pos[2] - atpos[2]), 2), 0.5)
        if not theta == 0:
            D1 = np.sin(theta) * np.cos(phi) * self.pos[1]
            D2 = np.sin(theta) * np.sin(phi) * self.pos[0]
            pha = k * (D - D1 - D2)
        else:
            D1 = np.sin(phi) * np.cos(theta) * self.pos[0]
            D2 = np.sin(phi) * 0 * self.pos[1]
            pha = k * (D - D1 - D2)

        return np.exp(1j * pha)

    def genIncidencePhase(self):
        k = 2 * np.pi * self.freq / 3e8
        atpos = [  # [x1,y1,z1] is the matrix of AP's position
            (np.ones((self.IRS_scale, self.IRS_scale)) * self.AP_pos[0]),  # x1
            (np.ones((self.IRS_scale, self.IRS_scale)) * self.AP_pos[1]),  # y1
            (np.ones((self.IRS_scale, self.IRS_scale)) * self.AP_pos[2])]  # z1
        D = np.power(np.power((self.pos[0] - atpos[0]), 2) + \
                     np.power((self.pos[1] - atpos[1]), 2) + \
                     np.power((self.pos[2] - atpos[2]), 2), 0.5)

        tmp = np.exp(1j * k * D)
        return np.angle(tmp)  # turn phase into [-pi, pi]


if __name__ == "__main__":
    K = 10
    User_number = 2
    AP_antenna_number = 1
    IRS_scale = 16
    interval = 0.03
    IRS_pos = np.array([interval / 2, interval / 2, 0])
    AP_pos = np.array([IRS_scale * interval / 2, IRS_scale * interval / 2, 4.5])

    tmp = []
    for i in range(User_number):
        rdn1 = (np.random.rand() * 2 - 1) * np.pi / 3
        rdn2 = (np.random.rand() * 2 - 1) * np.pi / 3
        height = np.random.randint(1, 2)
        tmp.append([IRS_scale * interval / 2 + height * np.tan(rdn1),
                    IRS_scale * interval / 2 + height * np.tan(rdn2), height])
    Usr_pos = np.array(tmp)

    chnl = clustered_SV_channel(interval, 16, IRS_pos, AP_pos, 1, User_number, AP_antenna_number)

    chnl.genLoS(Usr_pos)
    Psi = np.exp(1j * pi * np.ones((IRS_scale ** 2,)))
    H = chnl.genChnl(Psi, K)