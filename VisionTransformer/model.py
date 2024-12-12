import torch
from einops.layers.torch import Rearrange
from torch import Tensor
from torch import nn


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            Rearrange("b e h w -> b (h w) e"),
        )

        self.cls_token = nn.Parameter(torch.rand((1, 1, embed_dim)))
        self.positions = nn.Parameter(
            torch.rand((img_size // patch_size) ** 2 + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        x = self.projection(x)
        cls_tokens = torch.cat([self.cls_token.repeat(b, 1, 1), x], dim=1)

        x = cls_tokens + self.positions
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Linear Layers
        self.linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=8, qkv_bias=False, attn_drop=0.0, out_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scale = head_dim**-0.5

        self.qkv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias),
            Rearrange("b c (e h hd) -> b e h c hd", e=3, h=num_heads, hd=head_dim),
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Sequential(
            Rearrange("b h c hd -> b c (h hd)"), nn.Linear(embed_dim, embed_dim)
        )
        self.out_drop = nn.Dropout(out_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.qkv_rearrange = Rearrange("b e h c hd -> b (e h) c hd")

    def forward(self, x):
        # Attention
        x = self.qkv(x)

        v, q, k = torch.split(x, 1, dim=1)
        v = self.qkv_rearrange(v)
        q = self.qkv_rearrange(q)
        k = self.qkv_rearrange(k)
        x = self.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale)
        x = torch.matmul(x, v)
        # Out projection

        x = self.out(x)
        x = self.out_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.0, qkv_bias=False):
        super().__init__()

        # Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Attention
        self.attn = Attention(
            embed_dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            out_drop=drop_rate,
        )

        # Dropout
        self.drop = nn.Dropout(drop_rate)

        # MLP
        self.mlp = MLP(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            out_features=dim,
            drop=drop_rate,
        )

    def forward(self, x):
        y = self.norm1(x)
        # Attention
        y = self.attn(y)
        y = self.drop(y)
        x = x + y

        y = self.norm2(x)
        # MLP
        y = self.mlp(y)
        y = self.drop(y)
        return x + y


class Transformer(nn.Module):
    def __init__(
        self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.0, qkv_bias=False
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(dim, num_heads, mlp_ratio, drop_rate, qkv_bias=qkv_bias)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        drop_rate=0.0,
    ):
        super().__init__()

        # Присвоение переменных
        ...

        # Path Embeddings, CLS Token, Position Encoding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Transformer Encoder
        self.transformer = Transformer(
            depth=depth,
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )

        # Classifier
        self.classifier = nn.Sequential(nn.Linear(embed_dim, num_classes))

    def forward(self, x):

        # Path Embeddings, CLS Token, Position Encoding
        x = self.patch_embed(x)

        # Transformer Encoder
        x = self.transformer(x)

        # Classifier
        x = x[:, 0]
        x = self.classifier(x)

        return x
