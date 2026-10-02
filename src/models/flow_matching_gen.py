"""
Model 6: Flow Matching (OT-CFM) for PANCAN.

Adapted from scRNA-seq pipeline with all Run 1 fixes:
- StandardScaler normalization
- Proper continuous time embedding (no t*1000 hack)
- Mini-batch Optimal Transport coupling
- Zero-threshold post-processing
"""
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torchdiffeq
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class ContinuousTimeEmbedding(nn.Module):
    """Proper continuous time embedding for t in [0, 1]."""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1))


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, time_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.time_mlp = nn.Linear(time_dim, hidden_dim * 2)

    def forward(self, x, t):
        h = self.linear1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        time_emb = self.time_mlp(t)
        scale, shift = time_emb.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        return x + h


class MLPVelocity(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, time_dim=256, num_blocks=4):
        super().__init__()
        self.time_embed = ContinuousTimeEmbedding(time_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


def zero_threshold_postprocess(X_syn, X_train):
    """Match per-gene sparsity of training data."""
    P = X_train.shape[1]
    for j in range(P):
        real_zero_frac = np.mean(X_train[:, j] == 0)
        if real_zero_frac > 0.5:
            threshold = np.percentile(np.abs(X_syn[:, j]), real_zero_frac * 100)
            X_syn[:, j] = np.where(np.abs(X_syn[:, j]) < threshold, 0.0, X_syn[:, j])
    return X_syn


def generate(
    X_train: np.ndarray,
    n_samples: int,
    seed: int = 42,
    device: str = "cuda",
) -> np.ndarray:
    """
    Generate synthetic data using Optimal Transport Conditional Flow Matching.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    N, D = X_train.shape
    logger.info(f"PANCAN FlowMatching: N={N}, D={D}, device={device}")

    # StandardScaler normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    logger.info(f"  Applied StandardScaler: mean={np.mean(X_scaled):.4f}, std={np.std(X_scaled):.4f}")

    model = MLPVelocity(input_dim=D, hidden_dim=1024, time_dim=256, num_blocks=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    batch_size = min(64, N)
    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    max_epochs = 2000
    patience = 200
    best_loss = float("inf")
    patience_counter = 0
    sigma_min = 0.001

    logger.info(f"  Training: max_epochs={max_epochs}, batch_size={batch_size}, patience={patience}")
    model.train()
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            B = batch_x.shape[0]

            x_0 = torch.randn_like(batch_x)
            x_1 = batch_x

            # Optimal transport matching
            with torch.no_grad():
                dist = torch.cdist(x_0, x_1)
                row_ind, col_ind = linear_sum_assignment(dist.cpu().numpy())
                x_1 = x_1[col_ind]

            t = torch.rand(B, 1, device=device)

            # OT-CFM interpolation
            x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
            u_t = x_1 - (1 - sigma_min) * x_0

            v_pred = model(x_t, t.squeeze(-1))
            loss = nn.functional.mse_loss(v_pred, u_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * B

        epoch_loss /= N

        if epoch % 100 == 0 or epoch == max_epochs - 1:
            logger.info(f"  Epoch {epoch+1}/{max_epochs}: loss={epoch_loss:.6f} (best={best_loss:.6f})")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    logger.info("  Training complete. Starting generation...")
    model.eval()

    class ODEWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, t, x):
            t_tensor = torch.full((x.shape[0],), t.item(), device=x.device)
            return self.model(x, t_tensor)

    ode_wrapper = ODEWrapper(model)

    samples = []
    with torch.no_grad():
        batch_size_gen = min(256, n_samples)
        for i in range(0, n_samples, batch_size_gen):
            b = min(batch_size_gen, n_samples - i)
            x_0 = torch.randn((b, D), device=device)
            t_span = torch.linspace(0, 1, 100, device=device)

            traj = torchdiffeq.odeint(ode_wrapper, x_0, t_span, method="euler")
            x_1 = traj[-1]
            samples.append(x_1.cpu().numpy())

    generated_data = np.vstack(samples)

    # Inverse StandardScaler
    generated_data = scaler.inverse_transform(generated_data)
    generated_data = np.clip(generated_data, 0, None).astype(np.float32)

    # Zero-threshold post-processing
    generated_data = zero_threshold_postprocess(generated_data, X_train)

    logger.info(f"PANCAN FlowMatching: Generated {n_samples} samples, "
                f"sparsity={np.mean(generated_data == 0) * 100:.1f}%")
    return generated_data
