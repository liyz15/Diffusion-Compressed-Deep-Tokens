# This is a vanilla transformer encoder
# Adapted from https://github.com/openai/CLIP/blob/main/clip/model.py

from collections import OrderedDict
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        return super().forward(x)
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class TransformerWithInOutProjPos(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: int = 2,
        width: int = 1024,
        heads: int = 12,
        context_length: int = 16,  # number of frames
        causal: bool = False,  # causal mask in transformer
        zero_init: bool = False,
        ln_final: bool = True,
    ):
        super().__init__()

        self.context_length = context_length

        if causal:
            self.context_length = context_length
            mask = self.build_attention_mask()
        self.transformer = Transformer(width, layers, heads, mask if causal else None)

        self.input_proj = nn.Linear(input_dim, width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, width)
        )
        if ln_final:
            self.ln_final = LayerNorm(width)
        else:
            self.ln_final = Identity()
        self.output_proj = nn.Linear(width, output_dim)

        self._init_weights(zero_init=zero_init)

    def _init_weights(self, zero_init=False):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        nn.init.normal_(self.input_proj.weight, std=proj_std)
        nn.init.normal_(self.output_proj.weight, std=proj_std)

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        if zero_init:
            if not isinstance(self.ln_final, Identity):
                print("Zero init may not work as expected as the final LayerNorm is not Identity")
            # Input proj
            nn.init.eye_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

            # Positional embedding
            nn.init.zeros_(self.positional_embedding)

            # Transformer
            for block in self.transformer.resblocks:
                nn.init.zeros_(block.attn.out_proj.weight)
                nn.init.zeros_(block.attn.out_proj.bias)

                nn.init.zeros_(block.mlp.c_proj.weight)
                nn.init.zeros_(block.mlp.c_proj.bias)

            # Output proj
            nn.init.eye_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor):
        orig_x = x
        x = self.input_proj(x)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.output_proj(self.ln_final(x))
        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fake_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    # Test the model
    model = TransformerWithInOutProjPos(
        input_dim=1024, output_dim=1024, layers=4, width=1024, heads=16, 
        context_length=16, causal=False, zero_init=True, ln_final=False
    )

    x = torch.randn(2, 16, 1024)

    y = model(x)
