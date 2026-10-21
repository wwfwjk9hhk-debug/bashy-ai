import torch
import torch.nn as nn
from typing import List

class MarkovPrior(nn.Module):
    def __init__(self, vocab_size: int, smoothing: float = 1e-4):
        super().__init__()
        self.vocab_size, self.smoothing = vocab_size, smoothing
        self.register_buffer("transition_counts", torch.zeros(vocab_size, vocab_size))
        self.register_buffer("transition_logits", torch.zeros(vocab_size, vocab_size))
    def fit_from_token_ids(self, token_ids: List[int]):
        counts = torch.zeros(self.vocab_size, self.vocab_size)
        for prev, nxt in zip(token_ids[:-1], token_ids[1:]):
            if prev < self.vocab_size and nxt < self.vocab_size:
                counts[prev, nxt] += 1.0
        self.transition_counts.copy_(counts)
        probs = (counts + self.smoothing) / (counts + self.smoothing).sum(dim=-1, keepdim=True)
        self.transition_logits.copy_(torch.log(probs))
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transition_logits[input_ids]
