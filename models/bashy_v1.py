import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rmsnorm import RMSNorm
from models.swiglu import SwiGLUMLP
from models.attention import GroupedQueryAttention
from models.latent_bayesian import LatentBayesianModule
from models.markov_prior import MarkovPrior
from models.probabilistic_fusion import ProbabilisticFusion, sample_with_top_k_top_p

@dataclass
class BashyConfig:
    vocab_size: int = 16384
    embed_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2
    max_seq_len: int = 2048
    latent_dim: int = 64
    kl_weight: float = 0.005
    dropout: float = 0.0
    tie_word_embeddings: bool = True

class BashyBlock(nn.Module):
    def __init__(self, config: BashyConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim)
        self.attn = GroupedQueryAttention(config.embed_dim, config.num_heads, config.num_kv_heads, config.max_seq_len, config.dropout)
        self.mlp_norm = RMSNorm(config.embed_dim)
        self.mlp = SwiGLUMLP(config.embed_dim)
    def forward(self, x, start_pos: int = 0, latent_mod=None, kv_cache=None, use_cache=False):
        if latent_mod is not None: x = x + latent_mod
        attn_out, new_cache = self.attn(self.attn_norm(x), start_pos=start_pos, kv_cache=kv_cache, use_cache=use_cache)
        h = x + attn_out
        out = h + self.mlp(self.mlp_norm(h))
        return out, new_cache

class BashyV1LM(nn.Module):
    def __init__(self, config: Optional[BashyConfig] = None):
        super().__init__()
        self.config = config or BashyConfig()
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.embed_dim)
        self.latent_module = LatentBayesianModule(self.config.embed_dim, self.config.latent_dim)
        self.markov_prior = MarkovPrior(self.config.vocab_size)
        self.layers = nn.ModuleList([BashyBlock(self.config) for _ in range(self.config.num_layers)])
        self.norm = RMSNorm(self.config.embed_dim)
        self.fusion = ProbabilisticFusion(self.config.embed_dim, self.config.vocab_size)
        if self.config.tie_word_embeddings:
            self.fusion.lm_head.weight = self.tok_embeddings.weight
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def fit_markov_corpus(self, token_ids): self.markov_prior.fit_from_token_ids(token_ids)
    def forward(self, input_ids, targets=None, start_pos: int = 0, kv_caches=None, use_cache=False):
        x = self.tok_embeddings(input_ids)
        latent_mod, kl_loss, _ = self.latent_module(x)
        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            c = kv_caches[i] if kv_caches else None
            x, nc = layer(x, start_pos=start_pos, latent_mod=latent_mod, kv_cache=c, use_cache=use_cache)
            if use_cache: new_caches.append(nc)
        x = self.norm(x)
        markov_logits = self.markov_prior(input_ids)
        fused_logits, alpha_gate = self.fusion(x, markov_logits=markov_logits)
        total_loss = None
        metrics = {"kl_loss": kl_loss.item() if kl_loss is not None else 0.0, "alpha_mean": alpha_gate.mean().item()}
        if targets is not None:
            ce_loss = F.cross_entropy(fused_logits.view(-1, self.config.vocab_size), targets.view(-1), ignore_index=-100)
            total_loss = ce_loss + self.config.kl_weight * kl_loss
            metrics["ce_loss"] = ce_loss.item()
            metrics["total_loss"] = total_loss.item()
        return fused_logits, total_loss, metrics, new_caches
    @torch.no_grad()
    def generate_with_kv_cache(self, prompt_ids, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9, eos_token_id=None, device=None):
        self.eval()
        if device is None: device = next(self.parameters()).device
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = list(prompt_ids)
        logits, _, _, kv_caches = self(inp, start_pos=0, use_cache=True)
        nxt = sample_with_top_k_top_p(logits[0, -1, :], temperature=temperature, top_k=top_k, top_p=top_p).item()
        generated.append(nxt)
        start_pos = len(prompt_ids)
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and nxt == eos_token_id: break
            step_in = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, _, _, kv_caches = self(step_in, start_pos=start_pos, kv_caches=kv_caches, use_cache=True)
            nxt = sample_with_top_k_top_p(logits[0, -1, :], temperature=temperature, top_k=top_k, top_p=top_p).item()
            generated.append(nxt)
            start_pos += 1
        return generated
