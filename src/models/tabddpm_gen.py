"""
Model 5: TabDDPM — Denoising Diffusion for PANCAN.

Adapted from scRNA-seq pipeline with all Run 1 fixes:
- StandardScaler normalization (zero-mean, unit-variance)
- Cosine beta schedule (Nichol & Dhariwal, 2021)
- MLP denoiser with residual blocks + FiLM conditioning
- Zero-threshold post-processing
"""
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


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


class MLPDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, time_dim=256, num_blocks=4):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
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


def cosine_beta_schedule(T, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal, 2021)."""
    steps = np.arange(T + 1, dtype=np.float64) / T
    alphas_bar = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.tensor(np.clip(betas, 0.0001, 0.999), dtype=torch.float32)


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
    Generate synthetic data using TabDDPM (Denoising Diffusion).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    N, D = X_train.shape
    logger.info(f"PANCAN TabDDPM: N={N}, D={D}, device={device}")

    # StandardScaler normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    logger.info(f"  Applied StandardScaler: mean={np.mean(X_scaled):.4f}, std={np.std(X_scaled):.4f}")

    # Diffusion parameters — cosine schedule
    T = 1000
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    model = MLPDenoiser(input_dim=D, hidden_dim=1024, time_dim=256, num_blocks=4).to(device)
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

    logger.info(f"  Training: max_epochs={max_epochs}, batch_size={batch_size}, patience={patience}")
    model.train()
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            B = batch_x.shape[0]

            t = torch.randint(0, T, (B,), device=device).long()
            noise = torch.randn_like(batch_x)

            alpha_bar_t = alphas_cumprod[t].unsqueeze(-1)
            x_t = torch.sqrt(alpha_bar_t) * batch_x + torch.sqrt(1 - alpha_bar_t) * noise

            pred_noise = model(x_t, t.float())
            loss = nn.functional.mse_loss(pred_noise, noise)

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

    samples = []
    with torch.no_grad():
        batch_size_gen = min(256, n_samples)
        for i in range(0, n_samples, batch_size_gen):
            b = min(batch_size_gen, n_samples - i)
            x = torch.randn((b, D), device=device)

            for t_step in reversed(range(T)):
                t_tensor = torch.full((b,), t_step, device=device, dtype=torch.float32)
                pred_noise = model(x, t_tensor)

                alpha = alphas[t_step]
                alpha_bar = alphas_cumprod[t_step]
                beta = betas[t_step]

                if t_step > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
                )
                x = x + torch.sqrt(beta) * noise

            samples.append(x.cpu().numpy())

    generated_data = np.vstack(samples)

    # Inverse StandardScaler
    generated_data = scaler.inverse_transform(generated_data)
    generated_data = np.clip(generated_data, 0, None).astype(np.float32)

    # Zero-threshold post-processing
    generated_data = zero_threshold_postprocess(generated_data, X_train)

    logger.info(f"PANCAN TabDDPM: Generated {n_samples} samples, "
                f"sparsity={np.mean(generated_data == 0) * 100:.1f}%")
    return generated_data
