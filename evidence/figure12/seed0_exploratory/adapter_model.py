"""Inference-only Modest-Align adapters; frozen encoders are not distributed."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.6):
        super().__init__()
        self.fn = nn.Sequential(nn.Linear(dim, expansion_factor * dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(expansion_factor * dim, dim))
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.fn(self.ln(x))


class AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_proj = nn.Sequential(*[Block(1536) for _ in range(4)],
                                      nn.LayerNorm(1536), nn.Linear(1536, 512))
        self.text_proj = nn.Sequential(*[Block(1024) for _ in range(4)],
                                       nn.LayerNorm(1024), nn.Linear(1024, 512))
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(self, image_features, text_features):
        image = F.normalize(self.vis_proj(image_features), dim=-1)
        text = F.normalize(self.text_proj(text_features), dim=-1)
        return image @ text.T


def load_adapter(path):
    from safetensors.torch import load_file
    model = AdapterModel().float()
    model.load_state_dict(load_file(str(Path(path)), device='cpu'), strict=True)
    model.eval()
    return model
