import torch
import torch.nn as nn
from typing import Tuple

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim, self.max_seq_len = dim, max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    def _build_cache(self, seq_len: int):
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.cat((torch.outer(pos, self.inv_freq), torch.outer(pos, self.inv_freq)), dim=-1)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
    def _rotate_half(self, x):
        return torch.cat((-x[..., self.dim//2:], x[..., :self.dim//2]), dim=-1)
    def forward(self, q, k, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]
        if start_pos + T > self.max_seq_len: self._build_cache(start_pos + T)
        cos = self.cos_cached[start_pos : start_pos + T].unsqueeze(0).unsqueeze(1).to(q.dtype).to(q.device)
        sin = self.sin_cached[start_pos : start_pos + T].unsqueeze(0).unsqueeze(1).to(q.dtype).to(q.device)
        return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (self._rotate_half(k) * sin)
