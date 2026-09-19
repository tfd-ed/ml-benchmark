"""Workloads shared between experiments: seeded data pipelines and the GPT training step."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .models import GPT, count_params

PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}

# --------------------------------------------------------------------------- token data (transformer / sustained / memory)


def make_token_data(vocab_size: int, num_sequences: int, length: int, branching: int, seed: int) -> torch.Tensor:
    """Learnable synthetic token streams: a fixed random Markov chain where every token has `branching`
    possible successors, so the achievable loss is ln(branching) (< ln(vocab)). Generated on the CPU
    with a seeded generator -> identical on every backend. Shape [num_sequences, length + 1]."""
    g = torch.Generator().manual_seed(seed)
    succ = torch.randint(0, vocab_size, (vocab_size, branching), generator=g)
    x = torch.randint(0, vocab_size, (num_sequences,), generator=g)
    out = torch.empty(num_sequences, length + 1, dtype=torch.long)
    out[:, 0] = x
    for t in range(1, length + 1):
        x = succ[x, torch.randint(0, branching, (num_sequences,), generator=g)]
        out[:, t] = x
    return out


@dataclass
class GPTWorkload:
    """A complete training workload: model + AdamW + data. `step(i)` performs one training step
    (host->device copy, forward, backward, optimizer) and returns the loss as a Python float."""

    model: torch.nn.Module
    opt: torch.optim.Optimizer
    data: torch.Tensor
    batch_size: int
    seq_len: int
    device: torch.device
    precision: str
    scaler: Any
    n_params: int
    scaler_note: str | None = None

    def batch(self, i: int) -> torch.Tensor:
        n = self.data.shape[0]
        idx = (torch.arange(self.batch_size) + i * self.batch_size) % n
        return self.data[idx, : self.seq_len + 1]

    def step(self, i: int) -> float:
        tokens = self.batch(i).to(self.device)
        x, y = tokens[:, :-1], tokens[:, 1:]
        amp_dtype = PRECISIONS[self.precision]
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = self.model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            self.opt.step()
        return float(loss.item())  # .item() also synchronises the device


def build_gpt_workload(model_cfg: dict, batch_size: int, seq_len: int, device: torch.device, seed: int, lr: float,
                       precision: str = "fp32", num_sequences: int = 2048, branching: int = 4,
                       max_len: int | None = None) -> GPTWorkload:
    """Model/data are created on the CPU from `seed`, then moved once to `device`."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision}")
    torch.manual_seed(seed)
    model = GPT(model_cfg["vocab_size"], model_cfg["d_model"], model_cfg["n_layers"], model_cfg["n_heads"],
                max_len or seq_len, model_cfg.get("mlp_ratio", 4))
    n_params = count_params(model)
    model = model.to(device)
    data = make_token_data(model_cfg["vocab_size"], max(num_sequences, batch_size), seq_len, branching, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler, note = None, None
    if precision == "fp16":
        scaler = torch.amp.GradScaler(device.type, enabled=True)
        if not scaler.is_enabled():  # never train fp16 without loss scaling silently
            raise NotImplementedError(f"GradScaler is not supported on '{device.type}'; fp16 training would be unscaled")
    return GPTWorkload(model, opt, data, batch_size, seq_len, device, precision, scaler, n_params, note)


# --------------------------------------------------------------------------- Fashion-MNIST (CNN)

FMNIST_MEAN = 0.2860
FMNIST_STD = 0.3530


def load_fashion_mnist(data_dir: Path, train_samples: int | None, val_samples: int | None, seed: int):
    """Returns normalised float tensors (train_x, train_y, val_x, val_y) on the CPU, shaped N,3,32,32.
    The 28x28 grayscale images are zero-padded to 32x32 and replicated over 3 channels so the CIFAR-style
    ResNet (and its FLOPs) is unchanged. The subsets are a seeded random selection, identical for every backend."""
    from torchvision.datasets import FashionMNIST

    def prep(ds, n):
        x, y = ds.data, ds.targets.long()  # uint8 N,28,28
        if n is not None and n < len(y):
            idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))[:n]
            x, y = x[idx], y[idx]
        x = F.pad(x, (2, 2, 2, 2)).unsqueeze(1).expand(-1, 3, -1, -1)
        return ((x.float() / 255.0) - FMNIST_MEAN) / FMNIST_STD, y

    train = FashionMNIST(root=str(data_dir), train=True, download=True)
    test = FashionMNIST(root=str(data_dir), train=False, download=True)
    (tx, ty), (vx, vy) = prep(train, train_samples), prep(test, val_samples)
    return tx.contiguous(), ty, vx.contiguous(), vy


def augment_epoch(x: torch.Tensor, y: torch.Tensor, seed: int, crop_padding: int, flip: bool):
    """Shuffle + random-crop (zero padding) + horizontal flip for a whole epoch, on the CPU with a seeded
    generator, so every backend receives bit-identical batches in the same order."""
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    perm = torch.randperm(n, generator=g)
    x, y = x[perm], y[perm]
    if crop_padding > 0:
        p = crop_padding
        padded = F.pad(x, (p, p, p, p))
        off = torch.randint(0, 2 * p + 1, (n, 2), generator=g)
        x = torch.stack([padded[i, :, off[i, 0]:off[i, 0] + 32, off[i, 1]:off[i, 1] + 32] for i in range(n)])
    if flip:
        mask = torch.rand(n, generator=g) < 0.5
        x = torch.where(mask.view(-1, 1, 1, 1), x.flip(3), x)
    return x.contiguous(), y
