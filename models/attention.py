import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from models.rope import RotaryPositionalEmbedding

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, num_kv_heads: int = 2, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.embed_dim, self.num_heads, self.num_kv_heads = embed_dim, num_heads, num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
        self.dropout = dropout
    def _repeat_kv(self, x, n_rep):
        if n_rep == 1: return x
        B, H, T, D = x.shape
        return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)
    def forward(self, x, start_pos: int = 0, kv_cache=None, use_cache=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, start_pos=start_pos)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v) if use_cache else None
        k_rep = self._repeat_kv(k, self.num_queries_per_kv)
        v_rep = self._repeat_kv(v, self.num_queries_per_kv)
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=(T > 1), dropout_p=drop_p)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)), new_cache
