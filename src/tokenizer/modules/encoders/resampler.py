# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
import math
import torch
import torch.nn as nn

from collections import OrderedDict
from einops import rearrange


class ImageProjModel(nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        #embeds = image_embeds
        embeds = image_embeds.type(list(self.proj.parameters())[0].dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class ResidualSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, )
        self.ln = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln(x))
        return x


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        video_length=None, # using frame-wise version or not
        output_shape=None,
        extra_self_attn=False,
        extra_output_proj=False,
        use_cosine=False,
        use_norm_out=True,
    ):
        super().__init__()
        ## queries for a single frame / image
        self.num_queries = num_queries 
        self.video_length = video_length
        self.output_shape = output_shape

        ## <num_queries> queries for each frame
        if video_length is not None: 
            raise NotImplementedError("video_length is not None, this will result in num_queries * video_length queries")
            num_queries = num_queries * video_length

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        if use_norm_out:
            self.norm_out = nn.LayerNorm(output_dim)
        else:
            self.norm_out = nn.Identity()

        if extra_self_attn:
            # For the convenience of using nn.MultiheadAttention, here inner dim is not changed
            num_self_attn_heads = dim // dim_head
            assert dim % dim_head == 0, f"dim {dim} must be divisible by dim_head {dim_head}"
            self.self_attn_layers = nn.ModuleList([
                ResidualSelfAttention(dim, num_self_attn_heads) for _ in range(depth)
            ])
            for l in self.self_attn_layers:
                nn.init.zeros_(l.attn.out_proj.weight)
                nn.init.zeros_(l.attn.out_proj.bias)
        else:
            self.self_attn_layers = nn.ModuleList([
                nn.Identity() for _ in range(depth)
            ])
        
        self.extra_output_proj = extra_output_proj
        if self.extra_output_proj:
            self.final_proj_out = nn.Linear(output_dim, output_dim)
            nn.init.eye_(self.final_proj_out.weight)
            nn.init.zeros_(self.final_proj_out.bias)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        if output_shape is not None:
            assert not extra_output_proj
            assert math.prod(output_shape) % num_queries == 0, \
                f"output shape must be divisible by num_queries, {math.prod(output_shape)} % {num_queries} != 0"
            final_output_dim = math.prod(output_shape) // num_queries

            self.final_proj_out = nn.Linear(output_dim, final_output_dim)
        
        self.use_cosine = use_cosine

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1) ## B (T L) C
        x = self.proj_in(x)
        
        for self_attn, (attn, ff) in zip(self.self_attn_layers, self.layers):
            latents = self_attn(latents)
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        latents = self.norm_out(latents) # B L C or B (T L) C

        if self.use_cosine:
            latents = nn.functional.normalize(latents, p=2, dim=-1)

        if self.output_shape != None:
            h = w = int(math.sqrt(latents.size(1)))  # 4, 4
            c, d1, d2 = self.output_shape  # 4, 32, 32
            d1 = d1 // h  # 8
            d2 = d2 // w  # 8

            latents = self.final_proj_out(latents)
            latents = rearrange(latents,
                                'b (h w) (c d1 d2) -> b c (h d1) (w d2)',
                                h=h, w=w, c=c, d1=d1, d2=d2)

        if self.extra_output_proj:
            latents = self.final_proj_out(latents)

        return latents


class MLP(nn.Module):
    def __init__(
            self, 
            in_dim, 
            hidden_dim, 
            out_dim, 
            depth, 
            post_norm=False, 
            zero_init_output=False, 
            add_shortcut=False,
        ):
        super().__init__()
        self.layers = nn.ModuleList([])

        # Input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        # Hidden layers
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        if zero_init_output:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

        if post_norm:
            self.post_norm = nn.LayerNorm(out_dim)
        else:
            self.post_norm = nn.Identity()
        
        self.add_shortcut = add_shortcut
        if self.add_shortcut:
            self.shortcut_proj = nn.Linear(in_dim, out_dim)
            nn.init.eye_(self.shortcut_proj.weight)
            nn.init.zeros_(self.shortcut_proj.bias)

    def forward(self, x):
        shortcut = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on the last layer
                x = nn.functional.gelu(x)

        if self.add_shortcut:
            x = x + self.shortcut_proj(shortcut)
        
        x = self.post_norm(x)
        
        return x
