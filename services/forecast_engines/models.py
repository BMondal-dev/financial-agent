"""Model abstraction layer for XGBoost and LSTM regressors.

Provides a unified interface for both model types with identical fit/predict API,
enabling fair benchmarking under the same data splits and evaluation protocol.
"""

from __future__ import annotations

from typing import Protocol, Literal
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Regressor(Protocol):
    """Protocol defining the common interface for all regression models."""

    def fit(self, X_train: np.ndarray | pd.DataFrame, y_train: np.ndarray | pd.Series) -> None:
        """Fit the model on training data."""
        ...

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generate predictions for input features."""
        ...


class XGBModel:
    """XGBoost regressor wrapper with fixed hyperparameters for fair comparison."""

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, X_train: np.ndarray | pd.DataFrame, y_train: np.ndarray | pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


if TORCH_AVAILABLE:
    class _LSTMNetwork(nn.Module):
        """PyTorch LSTM network for sequence regression."""

        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden).squeeze(-1)


    class LSTMModel:
        """LSTM regressor wrapper using tabular data reshaped into sequences.

        Uses a sliding window approach to create sequences from tabular features,
        matching the lag structure already present in the feature engineering.
        The sequence length defaults to 5 to align with the 5 lag features.
        """

        def __init__(
            self,
            seq_len: int = 5,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            learning_rate: float = 1e-3,
            epochs: int = 100,
            batch_size: int = 32,
            patience: int = 10,
            random_state: int = 42,
        ):
            self.seq_len = seq_len
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.patience = patience
            self.random_state = random_state

            self.model: _LSTMNetwork | None = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_mean: np.ndarray | None = None
            self.feature_std: np.ndarray | None = None
            self.n_features: int = 0

        def _to_numpy(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
            if isinstance(X, pd.DataFrame):
                return X.values.astype(np.float32)
            return X.astype(np.float32)

        def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
            if fit:
                self.feature_mean = X.mean(axis=0)
                self.feature_std = X.std(axis=0) + 1e-8
            return (X - self.feature_mean) / self.feature_std

        def _create_sequences(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
            """Create sequences for LSTM from tabular data using sliding window.

            Since features already include lags (target_lag_1 through target_lag_5),
            we create sequences by taking consecutive rows. This captures temporal
            patterns while avoiding look-ahead bias (all lags are properly shifted).
            """
            n_samples = len(X)
            if n_samples < self.seq_len:
                if y is not None:
                    return np.empty((0, self.seq_len, X.shape[1])), np.empty(0)
                return np.empty((0, self.seq_len, X.shape[1])), None

            sequences = []
            targets = [] if y is not None else None

            for i in range(self.seq_len - 1, n_samples):
                seq = X[i - self.seq_len + 1 : i + 1]
                sequences.append(seq)
                if y is not None:
                    targets.append(y[i])

            X_seq = np.array(sequences, dtype=np.float32)
            y_seq = np.array(targets, dtype=np.float32) if targets is not None else None
            return X_seq, y_seq

        def fit(self, X_train: np.ndarray | pd.DataFrame, y_train: np.ndarray | pd.Series) -> None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            X_np = self._to_numpy(X_train)
            y_np = y_train.values if isinstance(y_train, pd.Series) else y_train
            y_np = y_np.astype(np.float32)

            X_norm = self._normalize(X_np, fit=True)
            self.n_features = X_np.shape[1]

            X_seq, y_seq = self._create_sequences(X_norm, y_np)

            if len(X_seq) < 10:
                self.model = None
                return

            val_split = int(len(X_seq) * 0.85)
            X_train_seq, X_val_seq = X_seq[:val_split], X_seq[val_split:]
            y_train_seq, y_val_seq = y_seq[:val_split], y_seq[val_split:]

            self.model = _LSTMNetwork(
                input_size=self.n_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            X_train_t = torch.from_numpy(X_train_seq).to(self.device)
            y_train_t = torch.from_numpy(y_train_seq).to(self.device)
            X_val_t = torch.from_numpy(X_val_seq).to(self.device)
            y_val_t = torch.from_numpy(y_val_seq).to(self.device)

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            for epoch in range(self.epochs):
                self.model.train()
                n_batches = max(1, len(X_train_t) // self.batch_size)
                indices = torch.randperm(len(X_train_t))

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(X_train_t))
                    batch_indices = indices[start_idx:end_idx]

                    X_batch = X_train_t[batch_indices]
                    y_batch = y_train_t[batch_indices]

                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                if len(X_val_t) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_val_t)
                        val_loss = criterion(val_outputs, y_val_t).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            break

            if best_state is not None:
                self.model.load_state_dict(best_state)
            self.model.to(self.device)

        def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
            if self.model is None:
                return np.zeros(len(X))

            X_np = self._to_numpy(X)
            X_norm = self._normalize(X_np, fit=False)
            X_seq, _ = self._create_sequences(X_norm)

            if len(X_seq) == 0:
                return np.zeros(len(X))

            self.model.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(X_seq).to(self.device)
                preds = self.model(X_t).cpu().numpy()

            full_preds = np.zeros(len(X))
            full_preds[self.seq_len - 1 :] = preds
            full_preds[: self.seq_len - 1] = preds[0] if len(preds) > 0 else 0

            return full_preds


else:
    class LSTMModel:
        """Fallback LSTM model when PyTorch is not available."""

        def __init__(self, **kwargs):
            raise ImportError(
                "PyTorch is required for LSTM models. "
                "Install with: pip install torch"
            )

        def fit(self, X_train, y_train):
            pass

        def predict(self, X):
            pass


ModelType = Literal["xgb", "lstm"]


def get_model(model_type: ModelType = "xgb", **kwargs) -> XGBModel | LSTMModel:
    """Factory function to create a model instance.

    Args:
        model_type: Either "xgb" for XGBoost or "lstm" for LSTM.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        A model instance implementing the Regressor protocol.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type == "xgb":
        return XGBModel(**kwargs)
    elif model_type == "lstm":
        return LSTMModel(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'xgb' or 'lstm'.")
