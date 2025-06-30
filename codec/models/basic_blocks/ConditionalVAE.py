import torch
from .BASE_VAE import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar

from semantic_model.module_original.encoder_seq_shared_only import Encoder_PE
from semantic_model.module_original.decoder_seq_shared_only import Decoder_PE
from semantic_model.basic_blocks.RCB import RCB
from semantic_model.basic_blocks.RTCB import RTCB

Tensor = TypeVar('torch.tensor')


class ConditionalVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 compressed_channel: int,
                 img_size: int = 64,
                 feature_scale: float = 1.0,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.img_size = img_size
        self.compressed_channel = compressed_channel
        self.compressed_size = self.img_size//8

        self.embed_class = nn.Linear(embedding_dim, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, 3, kernel_size=1)

        # Build Encoder
        self.encoder = Encoder_PE(compressed_channel=self.compressed_channel, input_channel=3, device=kwargs['device'])
        feature_dim = self.compressed_channel*self.compressed_size**2
        self.latent_dim = int(feature_dim * feature_scale)
        self.fc_mu = nn.Linear(feature_dim, self.latent_dim)
        self.fc_var = nn.Linear(feature_dim, self.latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(self.latent_dim, feature_dim)
        self.decoder = Decoder_PE(compressed_channel=compressed_channel, reconstruct_channel=3, device=kwargs['device'])

        # output layer
        self.final_layer = nn.Sequential(
                            nn.BatchNorm2d(3),
                            nn.LeakyReLU(),
                            nn.Conv2d(3,
                                      out_channels=3,
                                      kernel_size=3,
                                      padding=1),
                            nn.Tanh())

    def encode(self, input: Tensor, sub_CSI: Tensor=None) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input, sub_CSI)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor, sub_CSI: Tensor=None) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.compressed_channel, self.compressed_size, self.compressed_size)
        result = self.decoder(result, sub_CSI)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, sub_CSI: Tensor=None, **kwargs) -> List[Tensor]:
        y = kwargs['labels']
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x, sub_CSI)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim=1)
        return [self.decode(z, sub_CSI), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]