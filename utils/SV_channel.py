import numpy as np
import random
import datetime
import pdb
from math import *

random.seed(0)
np.random.seed(0)


class Saleh_Valenzuela_Channel():
    def __init__(self, IRS_scale, IRS_pos, AP_pos, userNum, antennaNum):
        self.N = IRS_scale ** 2
        self.scale = int(self.N ** 0.5)
        self.userNum = userNum
        self.antNum = antennaNum
        self.IRS_pos = IRS_pos
        self.AP_pos = AP_pos

        self.H_B2R_LoS = np.zeros((self.antNum, self.N), dtype=complex)
        self.H_R2U_LoS = np.zeros((self.N, self.userNum), dtype=complex)
        self.H_B2U_LoS = np.zeros((self.antNum, self.userNum), dtype=complex)

    def steering_vec(self, angle, antNum, lambda_scale=None):
        if lambda_scale is None:
            sv = np.exp(1j * angle * pi * np.arange(0, antNum, 1))
            sv = sv.reshape(-1, 1)
        else:
            sv = np.exp(1j * angle * (pi / lambda_scale) * np.arange(0, antNum, 1))
            sv = sv.reshape(-1, 1)
        return sv

    def ChannelMdl(self, pos_A, pos_B, atScale_A, atScale_B):
        dis_AB = np.linalg.norm(pos_A - pos_B)  # distance
        n_AB = (pos_A - pos_B) / dis_AB  # direction vector
        angleA = [np.linalg.multi_dot([[1, 0, 0], n_AB]),
                  np.linalg.multi_dot([[0, 1, 0], n_AB]),
                  np.linalg.multi_dot([[0, 0, 1], n_AB])]

        sv_A = np.kron(self.steering_vec(angleA[0], atScale_A[0]), self.steering_vec(angleA[1], atScale_A[1]))
        sv_A = np.kron(sv_A, self.steering_vec(angleA[2], atScale_A[2]))
        angleB = [np.linalg.multi_dot([[1, 0, 0], n_AB]),
                  np.linalg.multi_dot([[0, 1, 0], n_AB]),
                  np.linalg.multi_dot([[0, 0, 1], n_AB])]
        sv_B = np.kron(self.steering_vec(angleB[0], atScale_B[0]), self.steering_vec(angleB[1], atScale_B[1]))
        sv_B = np.kron(sv_B, self.steering_vec(angleB[2], atScale_B[2]))

        H = np.linalg.multi_dot([sv_A, np.matrix.getH(sv_B)])
        return H

    # Non LoS channel
    def genNonLoS(self):
        H_B2U_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(self.antNum, self.userNum)) + 1j * np.random.normal(0, 1, size=(self.antNum, self.userNum)))
        H_B2R_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(self.antNum, self.N)) + 1j * np.random.normal(0, 1, size=(self.antNum, self.N)))
        H_R2U_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(self.N, self.userNum)) + 1j * np.random.normal(0, 1, size=(self.N, self.userNum)))
        return H_B2R_NLoS, H_R2U_NLoS, H_B2U_NLoS

    # LoS channel
    def genLoS(self, pos_User):
        atScale_User = [1, 1, 1]
        atScale_AP = [self.antNum, 1, 1]
        atScale_IRS = [self.scale, self.scale, 1]

        for iu in range(self.userNum):
            h_B2U_LoS = self.ChannelMdl(self.AP_pos, pos_User[iu], atScale_AP, atScale_User)
            self.H_B2U_LoS[:, iu] = h_B2U_LoS.reshape(-1)
            h_U2R_LoS = self.ChannelMdl(self.IRS_pos, pos_User[iu], atScale_IRS, atScale_User)
            self.H_R2U_LoS[:, iu] = h_U2R_LoS.reshape(-1)
        self.H_B2R_LoS = self.ChannelMdl(self.AP_pos, self.IRS_pos, atScale_AP, atScale_IRS)

        return self.H_B2R_LoS, self.H_R2U_LoS, self.H_B2U_LoS

    def genRician(self, pos_User, K=10):
        self.genLoS(pos_User)
        H_B2R_NLoS, H_R2U_NLoS, H_B2U_NLoS = self.genNonLoS()
        H_B2R = sqrt(K / (K + 1)) * self.H_B2R_LoS + sqrt(1 / (K + 1)) * H_B2R_NLoS
        H_R2U = sqrt(K / (K + 1)) * self.H_R2U_LoS + sqrt(1 / (K + 1)) * H_R2U_NLoS
        H_B2U = sqrt(K / (K + 1)) * self.H_B2U_LoS + sqrt(1 / (K + 1)) * H_B2U_NLoS
        return H_B2R, H_R2U, H_B2U

    def RicianRefresh(self, K=10):
        H_B2R_NLoS, H_R2U_NLoS, H_B2U_NLoS = self.genNonLoS()
        H_B2R = sqrt(K / (K + 1)) * self.H_B2R_LoS + sqrt(1 / (K + 1)) * H_B2R_NLoS
        H_R2U = sqrt(K / (K + 1)) * self.H_R2U_LoS + sqrt(1 / (K + 1)) * H_R2U_NLoS
        H_B2U = sqrt(K / (K + 1)) * self.H_B2U_LoS + sqrt(1 / (K + 1)) * H_B2U_NLoS
        return H_B2R, H_R2U, H_B2U

    def genRayleigh(self, pos_User):
        self.genLoS(pos_User)
        return self.H_B2R_LoS, self.H_R2U_LoS, self.H_B2U_LoS

    def genAWGN(self):
        return (np.zeros((self.antNum, self.N), dtype=complex),
                np.zeros((self.N, self.userNum), dtype=complex),
                np.ones((self.antNum, self.userNum), dtype=complex))

    def genU2R(self):
        return self.H_R2U_LoS

    def genU2R_Rician(self, K=10):
        _, H_R2U, _ = self.RicianRefresh(K)
        return H_R2U


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

    chnl = Saleh_Valenzuela_Channel(16, IRS_pos, AP_pos, User_number, AP_antenna_number)

    chnl.genLoS(Usr_pos)
    Psi = np.exp(1j * pi * np.ones((IRS_scale ** 2,)))
    H_B2R, H_R2U, H_B2U = chnl.genRayleigh(Usr_pos)
    H = np.linalg.multi_dot([H_B2R, np.diag(Psi), H_R2U]) + H_B2U
    print(np.abs(H_B2R))