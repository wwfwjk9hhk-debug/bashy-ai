import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = None, multiple_of: int = 64):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * embed_dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, embed_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
