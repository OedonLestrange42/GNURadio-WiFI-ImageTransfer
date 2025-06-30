import pdb

import torch
import math
import torchvision.transforms as transforms
from einops import rearrange
# import numpy as np
from torchvision.transforms import ToPILImage
from torch import nn


def denormalize(img):
    """

    :param img: tensor [b, c, w, h] or [c, w, h]
    :return: denormalized tensor
    """
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    img_norm = torch.zeros_like(img)
    if len(img.shape) == 4:
        img_norm[:, 0, :, :] = std[0] * img[:, 0, :, :] + mean[0]
        img_norm[:, 1, :, :] = std[1] * img[:, 1, :, :] + mean[1]
        img_norm[:, 2, :, :] = std[2] * img[:, 2, :, :] + mean[2]
    elif len(img.shape) == 3:
        img_norm[0] = std[0] * img[0] + mean[0]
        img_norm[1] = std[1] * img[1] + mean[1]
        img_norm[2] = std[2] * img[2] + mean[2]
    else:
        Exception("input must have 3 or 4 channel")

    return img_norm


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


class JSCE():
    def __init__(self, weight_path, img_size, compressed_channel, CSI_bound=30, codec='SOMA-DSCN', device=torch.device('cpu')):

        if codec == 'SOMA-DSCN':
            from .models.module_original.encoder_dscn import Encoder
            from .models.module_original.decoder_seq_shared_only import Decoder_PE as Decoder
        else:
            from .models.module_original.encoder_seq_shared_only import Encoder_PE as Encoder
            from .models.module_original.decoder_seq_shared_only import Decoder_PE as Decoder

        self.device = device
        self.tensor2PIL = ToPILImage()
        self.shared_encoder = Encoder(compressed_channel=compressed_channel,
                                      device=device)
        self.shared_decoder = Decoder(compressed_channel=compressed_channel,
                                      reconstruct_channel=3,
                                      device=device)

        self.img2tensor = transforms.Compose([
                                                transforms.RandomGrayscale(),
                                                transforms.Resize((img_size[0], img_size[1])),  # 调整图片大小以匹配AlexNet结构
                                                transforms.ToTensor(),  # 将图片转换为Tensor
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
                                            ])

        self.CSI_bound = CSI_bound
        self.PE = positionalencoding2d(d_model=256, height=2 * self.CSI_bound, width=2 * self.CSI_bound).to(device)

        # init params
        checkpoint = torch.load(weight_path, map_location=device)
        shared_encoder_state_dict = {k[len('shared_encoder.'):]: v for k, v in checkpoint.items() if
                                     k.startswith('shared_encoder.')}
        shared_decoder_state_dict = {k[len('shared_decoder.'):]: v for k, v in checkpoint.items() if
                                     k.startswith('shared_decoder.')}

        self.shared_encoder.load_state_dict(shared_encoder_state_dict)
        self.shared_decoder.load_state_dict(shared_decoder_state_dict)

    def getCSI(self, key):
        real, imag = key.split('-')
        return torch.complex(torch.tensor([float(real)]), torch.tensor([float(imag)]))

    def img2msg(self, image_dict):
        """

        :param image_dict: {user_id: PIL.Image}
        :return: the merged latent feature
        """
        latent_feature = []
        for i, (key, img) in enumerate(image_dict.items()):
            CSI = self.getCSI(key)
            w = torch.clamp(torch.round(CSI.real[0]).detach() + self.CSI_bound, 0, 2 * self.CSI_bound - 1).to(torch.int)
            h = torch.clamp(torch.round(CSI.imag[0]).detach() + self.CSI_bound, 0, 2 * self.CSI_bound - 1).to(torch.int)
            label = self.PE[:, w, h].squeeze()
            # img = Image.open(img_path).convert('RGB')
            img_input = self.img2tensor(img).to(self.device).unsqueeze(0)
            img_latent = self.shared_encoder(img_input, label)
            latent_feature.append(img_latent)

        latent_signal = torch.mean(torch.stack(latent_feature), dim=0).squeeze(0).detach().cpu().numpy()
        latent_signal = rearrange(latent_signal, 'c w h -> w h c')
        return latent_signal

    def msg2img(self, latent_signal, user_id):
        """

        :param latent_signal: the merged latent feature
        :param user_id: user_id
        :return: the reconstructed image
        """
        if not isinstance(latent_signal, torch.Tensor):
            # Convert to tensor
            latent_signal = torch.tensor(latent_signal).to(self.device)
            latent_signal = rearrange(latent_signal, 'w h c -> c w h').unsqueeze(0)

        CSI = self.getCSI(user_id)
        w = torch.clamp(torch.round(CSI.real[0]).detach() + self.CSI_bound, 0, 2 * self.CSI_bound - 1).to(torch.int)
        h = torch.clamp(torch.round(CSI.imag[0]).detach() + self.CSI_bound, 0, 2 * self.CSI_bound - 1).to(torch.int)
        label = self.PE[:, w, h].squeeze()
        latent_signal = nn.functional.normalize(latent_signal, p=2, dim=1)
        img_reconstructed = self.shared_decoder(latent_signal, label).squeeze(0).detach().cpu()
        # img_reconstructed = np.clip(denormalize(img_reconstructed).detach().cpu().numpy(), 0, 1)
        # pdb.set_trace()
        img_reconstructed = self.tensor2PIL(torch.clamp(denormalize(img_reconstructed), 0, 1))
        return img_reconstructed
