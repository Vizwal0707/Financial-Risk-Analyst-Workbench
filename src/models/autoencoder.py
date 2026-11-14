"""PyTorch Deep Autoencoder for unsupervised financial anomaly detection.

Features:
- Symmetric bottleneck architecture with LeakyReLU, BatchNorm1d, and Dropout
- Trained exclusively on legitimate transactions (isFraud = 0)
- Validation-based early stopping
- Empirical percentile/CDF calibration on validation legitimate errors
- Reconstruction error feature attribution
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


class AutoencoderNet(nn.Module if TORCH_AVAILABLE else object):
    """Deep Symmetric Bottleneck Autoencoder Neural Network."""

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: List[int] = [64, 32, 16],
        decoder_hidden_dims: List[int] = [32, 64],
        dropout_rate: float = 0.10,
        leaky_relu_slope: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        encoder_layers = []
        in_d = input_dim
        for hidden_d in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(in_d, hidden_d))
            encoder_layers.append(nn.BatchNorm1d(hidden_d))
            encoder_layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
            encoder_layers.append(nn.Dropout(p=dropout_rate))
            in_d = hidden_d
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for hidden_d in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(in_d, hidden_d))
            decoder_layers.append(nn.BatchNorm1d(hidden_d))
            decoder_layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
            decoder_layers.append(nn.Dropout(p=dropout_rate))
            in_d = hidden_d
        decoder_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class DeepAutoencoder:
    """Wrapper managing training, evaluation, normalization, and inference."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        encoder_hidden_dims: List[int] = [64, 32, 16],
        decoder_hidden_dims: List[int] = [32, 64],
        dropout_rate: float = 0.10,
        leaky_relu_slope: float = 0.10,
        device: str = "auto",
        random_seed: int = 42,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepAutoencoder.")

        self.input_dim = input_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.dropout_rate = dropout_rate
        self.leaky_relu_slope = leaky_relu_slope
        self.random_seed = random_seed

        # Device selection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[AutoencoderNet] = None
        if input_dim is not None:
            self._build_model()

        # Calibration distribution of legitimate reconstruction errors
        self.val_legit_errors_sorted_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []

    def _build_model(self) -> None:
        torch.manual_seed(self.random_seed)
        self.model = AutoencoderNet(
            input_dim=self.input_dim,
            encoder_hidden_dims=self.encoder_hidden_dims,
            decoder_hidden_dims=self.decoder_hidden_dims,
            dropout_rate=self.dropout_rate,
            leaky_relu_slope=self.leaky_relu_slope,
        ).to(self.device)

    def compute_reconstruction_errors(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sample-level MSE and raw reconstructions."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        n_samples = len(X_arr)
        mse_errors = np.zeros(n_samples, dtype=np.float32)
        reconstructions = np.zeros_like(X_arr)

        dataset = TensorDataset(torch.from_numpy(X_arr))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            offset = 0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                recon = self.model(batch_x)
                sq_err = torch.mean((batch_x - recon) ** 2, dim=1)

                b_size = len(batch_x)
                mse_errors[offset:offset + b_size] = sq_err.cpu().numpy()
                reconstructions[offset:offset + b_size] = recon.cpu().numpy()
                offset += b_size

        return mse_errors, reconstructions

    # Convenience alias
    compute_reconstruction_error = compute_reconstruction_errors

    def calibrate_percentile_mapping(self, val_legit_errors: np.ndarray) -> None:
        """Construct empirical CDF from legitimate validation reconstruction errors."""
        self.val_legit_errors_sorted_ = np.sort(np.asarray(val_legit_errors, dtype=np.float32))

    def predict_anomaly_score(self, errors: np.ndarray) -> np.ndarray:
        """Map raw reconstruction errors to empirical percentiles in [0.0, 1.0]."""
        if self.val_legit_errors_sorted_ is None:
            # Fallback: min-max normalization
            e_min, e_max = np.min(errors), np.max(errors)
            return (errors - e_min) / (e_max - e_min + 1e-8)

        # Searchsorted to find empirical percentile
        ranks = np.searchsorted(self.val_legit_errors_sorted_, errors, side="right")
        percentiles = ranks / len(self.val_legit_errors_sorted_)
        return np.clip(percentiles, 0.0, 1.0)

    def save(self, filepath: str) -> None:
        """Save model checkpoint, architecture config, and calibration CDF."""
        if self.model is None:
            raise RuntimeError("Cannot save uninitialized model.")
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "encoder_hidden_dims": self.encoder_hidden_dims,
            "decoder_hidden_dims": self.decoder_hidden_dims,
            "dropout_rate": self.dropout_rate,
            "leaky_relu_slope": self.leaky_relu_slope,
            "val_legit_errors_sorted": self.val_legit_errors_sorted_,
            "feature_names": self.feature_names_,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = "auto") -> "DeepAutoencoder":
        """Load model checkpoint from disk."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepAutoencoder.")
        checkpoint = torch.load(filepath, map_location="cpu")
        instance = cls(
            input_dim=checkpoint["input_dim"],
            encoder_hidden_dims=checkpoint["encoder_hidden_dims"],
            decoder_hidden_dims=checkpoint["decoder_hidden_dims"],
            dropout_rate=checkpoint["dropout_rate"],
            leaky_relu_slope=checkpoint["leaky_relu_slope"],
            device=device,
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.val_legit_errors_sorted_ = checkpoint.get("val_legit_errors_sorted")
        instance.feature_names_ = checkpoint.get("feature_names", [])
        instance.model.eval()
        return instance


class AutoencoderTrainer:
    """Orchestrates training exclusively on legitimate transactions with early stopping."""

    def __init__(
        self,
        model: DeepAutoencoder,
        epochs: int = 25,
        batch_size: int = 2048,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        patience: int = 5,
        min_delta: float = 1e-4,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        X_train_legit: np.ndarray,
        X_val_legit: np.ndarray,
    ) -> Dict[str, Any]:
        """Train autoencoder exclusively on legitimate samples."""
        if self.model.model is None:
            self.model.input_dim = X_train_legit.shape[1]
            self.model._build_model()

        device = self.model.device
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_tensor = torch.from_numpy(X_train_legit.astype(np.float32))
        val_tensor = torch.from_numpy(X_val_legit.astype(np.float32))

        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor),
            batch_size=self.batch_size,
            shuffle=False,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.model.train()
            train_losses = []
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                recon = self.model.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation loss on legitimate holdout
            self.model.model.eval()
            val_losses = []
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(device)
                    recon = self.model.model(batch_x)
                    loss = criterion(recon, batch_x)
                    val_losses.append(loss.item())

            avg_train = float(np.mean(train_losses))
            avg_val = float(np.mean(val_losses))
            self.history_["train_loss"].append(avg_train)
            self.history_["val_loss"].append(avg_val)

            # Early stopping check
            if avg_val < best_val_loss - self.min_delta:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in self.model.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.model.load_state_dict(best_state)

        # Compute empirical calibration distribution on validation legitimate records
        val_errors, _ = self.model.compute_reconstruction_errors(X_val_legit)
        self.model.calibrate_percentile_mapping(val_errors)

        return {
            "epochs_trained": len(self.history_["train_loss"]),
            "best_val_loss": best_val_loss,
            "final_train_loss": self.history_["train_loss"][-1],
            "history": self.history_,
        }

