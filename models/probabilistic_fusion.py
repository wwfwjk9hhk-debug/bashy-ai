import torch
import torch.nn as nn
from typing import Optional, Tuple

class ProbabilisticFusion(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, default_alpha: float = 0.75):
        super().__init__()
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.gate_proj = nn.Linear(embed_dim, 1)
        nn.init.constant_(self.gate_proj.bias, 1.0)
    def forward(self, hidden_states: torch.Tensor, markov_logits: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        neural_logits = self.lm_head(hidden_states)
        if markov_logits is None: return neural_logits, torch.ones_like(neural_logits[..., :1])
        alpha = torch.sigmoid(self.gate_proj(hidden_states))
        return alpha * neural_logits + (1.0 - alpha) * markov_logits, alpha

def sample_with_top_k_top_p(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
    if temperature <= 0.0: return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = -float('Inf')
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
