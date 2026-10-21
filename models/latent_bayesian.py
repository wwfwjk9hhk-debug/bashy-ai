import torch
import torch.nn as nn
from typing import Tuple, Dict

class LatentBayesianModule(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int = 64):
        super().__init__()
        self.embed_dim, self.latent_dim = embed_dim, latent_dim
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.fc_shared = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        self.fc_proj = nn.Linear(latent_dim, embed_dim)
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        pooled = self.pooler(hidden_states.transpose(1, 2)).squeeze(-1)
        feat = self.fc_shared(pooled)
        mu = self.fc_mu(feat)
        logvar = torch.clamp(self.fc_logvar(feat), min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        latent_mod = self.fc_proj(z).unsqueeze(1)
        kl_div = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return latent_mod, kl_div.mean(), {"mu": mu, "logvar": logvar}
