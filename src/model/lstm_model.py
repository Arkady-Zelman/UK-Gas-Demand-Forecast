"""LSTM baseline for comparison with the XGBoost production model.

Included to demonstrate awareness of sequence models in the portfolio,
not because it outperforms XGBoost here — on tabular weather+demand data
with well-engineered lag features, gradient boosting typically wins.
The LSTM's main advantage would show up if we had richer intra-day data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.config import get
from src.features.engineer import FEATURE_COLS

logger = logging.getLogger(__name__)

SEQ_LEN: int = get("model.lstm.sequence_length", 14)
HIDDEN: int = get("model.lstm.hidden_size", 64)
N_LAYERS: int = get("model.lstm.num_layers", 1)
EPOCHS: int = get("model.lstm.epochs", 50)
BATCH: int = get("model.lstm.batch_size", 32)


class _LSTMNet(nn.Module):
    """Minimal LSTM -> Linear head for regression."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.1
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of length *seq_len* over (X, y) to produce 3-D tensors."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


class GasDemandLSTM:
    """Train / predict wrapper around the PyTorch LSTM."""

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        hidden: int = HIDDEN,
        n_layers: int = N_LAYERS,
        epochs: int = EPOCHS,
        batch_size: int = BATCH,
        lr: float = 1e-3,
    ) -> None:
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.net: _LSTMNet | None = None
        self.feature_cols = list(FEATURE_COLS)
        self.is_fitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        df: pd.DataFrame,
        target: str = "demand_mcm",
    ) -> "GasDemandLSTM":
        """Train the LSTM on a feature-engineered DataFrame."""
        X_raw = df[self.feature_cols].values.astype(np.float32)
        y_raw = df[target].values.astype(np.float32).reshape(-1, 1)

        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw).ravel()

        X_seq, y_seq = _make_sequences(X_scaled, y_scaled, self.seq_len)

        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        n_features = X_raw.shape[1]
        self.net = _LSTMNet(n_features, self.hidden, self.n_layers).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "LSTM epoch %d/%d  loss=%.4f",
                    epoch + 1, self.epochs, epoch_loss / len(dataset),
                )

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return demand predictions (mcm/d) for a feature-engineered DataFrame.

        Because the LSTM needs a warm-up window of *seq_len* rows, the output
        array is shorter than the input by *seq_len* rows.  The returned array
        corresponds to ``df.iloc[seq_len:]``.
        """
        if self.net is None:
            raise RuntimeError("Model not fitted")

        X_raw = df[self.feature_cols].values.astype(np.float32)
        X_scaled = self.scaler_X.transform(X_raw)
        X_seq, _ = _make_sequences(
            X_scaled, np.zeros(len(X_scaled)), self.seq_len
        )

        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            pred_scaled = self.net(t).cpu().numpy()

        pred = self.scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()
        return pred

    def predict_aligned(
        self, df: pd.DataFrame, target: str = "demand_mcm"
    ) -> pd.DataFrame:
        """Return a DataFrame with date, actual, and predicted columns.

        Aligned so that the warm-up rows are excluded.
        """
        preds = self.predict(df)
        aligned = df.iloc[self.seq_len :].copy()
        aligned = aligned[["date", target]].reset_index(drop=True)
        aligned["predicted"] = preds
        return aligned
