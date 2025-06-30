import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from semantic_model.basic_blocks.ViT import Transformer


class MAE(nn.Module):
    def __init__(
            self,
            *,
            encoder,
            decoder_dim,
            masking_ratio=0.75,
            decoder_depth=1,
            decoder_heads=8,
            decoder_dim_head=64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        self.pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, self.pixel_values_per_patch)

        # dynamic mid var
        self.device = None
        self.batch = None
        self.batch_range = None
        self.num_masked = None
        self.masked_patches = None
        self.unmasked_indices = None
        self.masked_indices = None

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, :num_patches]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)

            # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        # reconstruction = self.to_pixels(decoded_tokens).view(self.batch, 3, img_size, img_size)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss

    def encode(self, img, context):
        self.device = img.device

        # get patches

        patches = self.to_patch(img)
        self.batch, self.num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, :self.num_patches]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(self.device, dtype=tokens.dtype)

            # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        self.num_masked = int(self.masking_ratio * self.num_patches)
        rand_indices = torch.rand(self.batch, self.num_patches, device=self.device).argsort(dim=-1)
        self.masked_indices, self.unmasked_indices = rand_indices[:, :self.num_masked], rand_indices[:, self.num_masked:]

        # get the unmasked tokens to be encoded

        self.batch_range = torch.arange(self.batch, device=self.device)[:, None]
        tokens = tokens[self.batch_range, self.unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        self.masked_patches = patches[self.batch_range, self.masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens, context)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        return decoder_tokens

    def decode(self, decoder_tokens, context=None):
        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(self.unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=self.batch, n=self.num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(self.masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.zeros(self.batch, self.num_patches, self.decoder_dim, device=self.device)
        decoder_tokens[self.batch_range, self.unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[self.batch_range, self.masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens, context)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[self.batch_range, self.masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        img_size = int((self.num_patches * self.pixel_values_per_patch // 3) ** 0.5)
        reconstruction = self.to_pixels(decoded_tokens).view(self.batch, 3, img_size, img_size)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, self.masked_patches)
        return recon_loss, reconstruction
