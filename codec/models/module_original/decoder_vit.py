import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from semantic_model.basic_blocks.ViT import Transformer
from semantic_model.basic_blocks.Swin_ViT import PatchExpand, DimReducePatchExpand


class BasicLayer_Up(nn.Module):
    def __init__(self, img_size, patch_dim, depth=6, num_heads=16, norm_layer=nn.LayerNorm, upsample=None, device=torch.device('cpu')):
        super().__init__()
        # build blocks
        # patch merging layer
        if upsample is not None:
            self.upsample = upsample((img_size, img_size),
                                     dim=patch_dim,
                                     norm_layer=norm_layer).to(device)

            self.blocks = Transformer(dim=patch_dim // 2,
                                      depth=depth,
                                      heads=num_heads,
                                      dim_head=64,
                                      mlp_dim=512,
                                      dropout=0.).to(device)
        else:
            self.upsample = None
            self.blocks = Transformer(dim=patch_dim,
                                      depth=depth,
                                      heads=num_heads,
                                      dim_head=64,
                                      mlp_dim=512,
                                      dropout=0.).to(device)

    def forward(self, x, context=None):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.blocks(x, context)
        return x

    def flops(self):
        flops = 0
        flops += self.blocks.flops()
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops


class Decoder(nn.Module):
    def __init__(self, input_size, patch_dim, layer_depth=2, device=torch.device('cpu')):
        super().__init__()
        self.patch_dim = patch_dim
        self.backbone = nn.ModuleList([])
        for i in range(layer_depth):
            self.backbone.append(
                BasicLayer_Up(img_size=input_size * (2 ** i),
                              patch_dim=patch_dim // (i+1),
                              upsample=PatchExpand,
                              device=device
                )
            )

    def forward(self, x, context=None):
        for block in self.backbone:
            x = block(x, context=context[:, :, :x.shape[2] // 2])

        return x