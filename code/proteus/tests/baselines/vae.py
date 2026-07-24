"""Small reference VAE baseline for generative comparison."""

from __future__ import annotations

import numpy as np


def fit_vae(
    train: np.ndarray,
    latent_dim: int = 8,
    hidden_dim: int = 64,
    epochs: int = 50,
    batch_size: int = 128,
    seed: int = 0,
) -> dict:
    """Train a small VAE and return samples + reconstruction error.

    Returns a dict with keys: 'samples', 'recon_error', 'model'.
    Requires torch; raises ImportError if unavailable.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise ImportError("torch is required for the VAE baseline.") from e

    torch.manual_seed(seed)
    d = train.shape[1]

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.logvar = nn.Linear(hidden_dim, latent_dim)
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d),
            )

        def encode(self, x):
            h = self.enc(x)
            return self.mu(h), self.logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def decode(self, z):
            return self.dec(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tensor_data = torch.tensor(train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(recon, batch, reduction="sum")
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        recon, _, _ = model(tensor_data)
        recon_error = float(nn.functional.mse_loss(recon, tensor_data).item())
        z = torch.randn(train.shape[0], latent_dim)
        samples = model.decode(z).numpy()

    return {"samples": samples, "recon_error": recon_error, "model": model}
