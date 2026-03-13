
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LSTM Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):

    def __init__(
        self,
        input_size:   int,
        hidden_size:  int   = 128,
        num_layers:   int   = 2,
        dropout:      float = 0.2,
        output_size:  int   = 1,
    ) -> None:
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Multi-layer LSTM (dropout only applied between layers, not after last)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)

        # Forecast head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Predictions of shape (batch, output_size)
        """
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n:      (num_layers, batch, hidden_size)
        lstm_out, (h_n, _) = self.lstm(x)

        # Use the last hidden state of the top layer
        # h_n[-1]: (batch, hidden_size)
        last_hidden = h_n[-1]
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        return self.fc(last_hidden)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models.

    Adds fixed position-dependent signals to the input embeddings so
    the model can distinguish which time step each token corresponds to.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Args:
        d_model:  Model dimension (must match input feature size or projection)
        max_len:  Maximum sequence length supported
        dropout:  Dropout on the encoded output
    """

    def __init__(
        self,
        d_model: int,
        max_len: int   = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the (max_len, d_model) PE matrix once at init
        pe     = torch.zeros(max_len, d_model)
        pos    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div    = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])

        # Register as buffer (not a parameter — no gradients, but saved with model)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            x + positional encoding, same shape.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Transformer Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class TransformerForecaster(nn.Module):


    def __init__(
        self,
        input_size:          int,
        d_model:             int   = 64,
        nhead:               int   = 4,
        num_encoder_layers:  int   = 3,
        dim_feedforward:     int   = 256,
        dropout:             float = 0.1,
        output_size:         int   = 1,
        max_seq_len:         int   = 512,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model    = d_model
        self.input_size = input_size

        # Project raw features to d_model dimension
        self.input_projection = nn.Linear(input_size, d_model)

        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # (batch, seq, d_model) convention
            norm_first=True,     # Pre-LN: more stable training
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Forecast head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Predictions of shape (batch, output_size)
        """
        # Project features to d_model: (B, T, F) → (B, T, d_model)
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        # Transformer encoding: (B, T, d_model) → (B, T, d_model)
        enc = self.encoder(x)

        # Global average pooling over time dimension: (B, T, d) → (B, d)
        pooled = enc.mean(dim=1)
        pooled = self.dropout(pooled)

        return self.fc(pooled)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_type: str,
    input_size: int,
    cfg: dict,
) -> nn.Module:
    """
    Instantiate a model from configuration.

    Args:
        model_type: 'lstm' or 'transformer'
        input_size: Number of input features (from preprocessing)
        cfg:        The ml_models section of config.yaml

    Returns:
        Initialised (untrained) nn.Module.

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type == "lstm":
        m_cfg = cfg["lstm"]
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=m_cfg["hidden_size"],
            num_layers=m_cfg["num_layers"],
            dropout=m_cfg["dropout"],
        )
    elif model_type == "transformer":
        m_cfg = cfg["transformer"]
        model = TransformerForecaster(
            input_size=input_size,
            d_model=m_cfg["d_model"],
            nhead=m_cfg["nhead"],
            num_encoder_layers=m_cfg["num_encoder_layers"],
            dim_feedforward=m_cfg["dim_feedforward"],
            dropout=m_cfg["dropout"],
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            "Choose 'lstm' or 'transformer'."
        )
    return model