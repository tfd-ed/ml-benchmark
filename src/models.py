"""Model definitions. All models are built on the CPU from a fixed seed and then moved to the
device, so every backend starts from identical initial weights."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet-18 with the standard CIFAR adaptation (3x3 stride-1 stem, no max-pool)."""
    import torchvision

    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, mlp_ratio * d), nn.Linear(mlp_ratio * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.n_heads, c // self.n_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    """Small pre-LayerNorm decoder-only transformer with tied input/output embeddings."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_len: int, mlp_ratio: int = 4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, mlp_ratio) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x) @ self.tok.weight.T  # tied output projection
