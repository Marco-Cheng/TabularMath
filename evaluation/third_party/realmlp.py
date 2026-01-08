"""Standalone RealMLP TD-S style regressor used for TabularMath baselines.

The goal is to provide a lightweight, dependency-free approximation of the
RealMLP TD-S model described in the paper. The implementation is intentionally
simple but captures the main ingredients:
- LayerNorm-normalised residual MLP blocks with GELU activations
- Dropout inside the residual branches
- AdamW optimisation with cosine annealing warmup

This class exposes a scikit-learn compatible interface (`fit` / `predict`)
so it can be plugged into the existing experiment harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _ResidualMLPBlock(nn.Module):
    """Simple residual MLP block with LayerNorm + GELU."""

    def __init__(self, dim: int, hidden_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = dim * hidden_factor
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class _RealMLPBackbone(nn.Module):
    """Shallow RealMLP-inspired backbone with residual blocks."""

    def __init__(self, in_dim: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([_ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


@dataclass
class Standalone_RealMLP_TD_S_Regressor:
    """Minimal RealMLP TD-S style regressor with a sklearn-like interface."""

    hidden_dim: int = 512
    depth: int = 6
    dropout: float = 0.1
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 3e-4
    max_epochs: int = 200
    warmup_updates: int = 200
    device: str = "cpu"
    target_steps: int = 4000
    normalize_targets: bool = True

    def __post_init__(self) -> None:
        self._model: Optional[_RealMLPBackbone] = None
        self._device = torch.device(self.device)
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Standalone_RealMLP_TD_S_Regressor":
        if X.ndim != 2:
            raise ValueError("Input features must be 2D array-like.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        if self.normalize_targets:
            self._y_mean = float(y_tensor.mean())
            self._y_std = float(y_tensor.std(unbiased=False))
            if self._y_std == 0:
                self._y_std = 1.0
            y_tensor = (y_tensor - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0

        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = min(self.batch_size, len(dataset)) or 1
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        in_dim = X.shape[1]
        self._model = _RealMLPBackbone(in_dim, self.hidden_dim, self.depth, self.dropout).to(self._device)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_updates = max(1, self.target_steps // max(1, len(loader)))
        epochs = min(self.max_epochs, max(1, total_updates))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.lr * 0.05)

        self._model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                preds = self._model(xb)
                loss = torch.mean((preds - yb) ** 2)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        self._model.eval()
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self._device)
            preds = self._model(xb).cpu().numpy()
            if self.normalize_targets:
                preds = preds * self._y_std + self._y_mean
            return preds
