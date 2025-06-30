import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from semantic_model.basic_blocks.ViT import Transformer
from semantic_model.basic_blocks.Swin_ViT import PatchMerging, DimReducePatchExpand


class BasicLayer(nn.Module):
    def __init__(self, img_size, patch_dim, depth=6, num_heads=16, norm_layer=nn.LayerNorm, downsample=None, device=torch.device('cpu')):
        super().__init__()
        # build blocks
        self.blocks = Transformer(dim=patch_dim,
                                  depth=depth,
                                  heads=num_heads,
                                  dim_head=64,
                                  mlp_dim=512,
                                  dropout=0.).to(device)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample((img_size, img_size),
                                         dim=patch_dim,
                                         norm_layer=norm_layer).to(device)
        else:
            self.downsample = None

    def forward(self, x, context=None):
        x = self.blocks(x, context)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self):
        flops = 0
        flops += self.blocks.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class Encoder(nn.Module):
    def __init__(self, input_size, patch_dim, layer_depth=2, device=torch.device('cpu')):
        super().__init__()
        self.backbone = nn.ModuleList([])
        for i in range(layer_depth):
            self.backbone.append(
                BasicLayer(img_size=input_size//(2**i),
                           patch_dim=patch_dim*(i+1),
                           downsample=PatchMerging,
                           device=device
                )
            )

    def forward(self, x, context=None):
        for block in self.backbone:
            x = block(x, context=context[:, :, :x.shape[2]])

        return x