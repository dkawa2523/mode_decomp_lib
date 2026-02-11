"""Torch convolutional autoencoder decomposer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from mode_decomp_ml.config import cfg_get

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.optional import require_dependency

from .base import BaseDecomposer, _combine_masks, _normalize_mask, parse_bool, require_cfg
from mode_decomp_ml.plugins.registry import register_decomposer

try:  # optional dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch is optional
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

_MASK_POLICIES = {"error", "zero_fill"}
_NORMALIZE_OPTIONS = {"none", "standardize"}



def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        require_dependency(
            None,
            name="autoencoder decomposer",
            pip_name="torch",
        )


def _parse_int_list(value: Any, label: str) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [int(v) for v in value]
    elif isinstance(value, (int, np.integer)):
        items = [int(value)]
    else:
        raise ValueError(f"decompose.{label} must be int or list of int for autoencoder")
    if not items or any(item <= 0 for item in items):
        raise ValueError(f"decompose.{label} must contain positive ints for autoencoder")
    return items


def _activation_cls(name: str) -> type[nn.Module]:
    _require_torch()
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    if name not in mapping:
        raise ValueError(f"decompose.activation must be one of {tuple(mapping.keys())}, got {name}")
    return mapping[name]


if nn is not None:
    class _ConvAutoencoder(nn.Module):
        def __init__(
            self,
            *,
            in_channels: int,
            hidden_channels: Sequence[int],
            latent_dim: int,
            height: int,
            width: int,
            activation: str,
        ) -> None:
            super().__init__()
            act_cls = _activation_cls(activation)
            encoder_layers = []
            current = int(in_channels)
            for ch in hidden_channels:
                encoder_layers.append(nn.Conv2d(current, int(ch), kernel_size=3, padding=1))
                encoder_layers.append(act_cls())
                current = int(ch)
            self.encoder = nn.Sequential(*encoder_layers)
            self.flatten = nn.Flatten()
            self._encoded_channels = current
            self._height = int(height)
            self._width = int(width)
            self.fc_latent = nn.Linear(current * height * width, int(latent_dim))
            self.fc_decode = nn.Linear(int(latent_dim), current * height * width)

            decoder_layers = []
            rev_channels = list(reversed(hidden_channels))
            for idx, ch in enumerate(rev_channels):
                next_ch = rev_channels[idx + 1] if idx + 1 < len(rev_channels) else in_channels
                decoder_layers.append(nn.Conv2d(int(ch), int(next_ch), kernel_size=3, padding=1))
                if idx + 1 < len(rev_channels):
                    decoder_layers.append(act_cls())
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            x = self.encoder(x)
            x = self.flatten(x)
            return self.fc_latent(x)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            x = self.fc_decode(z)
            x = x.view(-1, self._encoded_channels, self._height, self._width)
            return self.decoder(x)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x))
else:  # pragma: no cover - torch is missing
    class _ConvAutoencoder:  # type: ignore[empty-body]
        pass


@register_decomposer("autoencoder")
class AutoencoderDecomposer(BaseDecomposer):
    """Torch convolutional autoencoder decomposer for grid domains."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        _require_torch()
        self.name = "autoencoder"

        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(
                f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}"
            )
        self._latent_dim = int(require_cfg(cfg, "latent_dim", label="decompose"))
        if self._latent_dim <= 0:
            raise ValueError("decompose.latent_dim must be > 0 for autoencoder")

        self._include_mean = parse_bool(cfg_get(cfg, "include_mean", None), default=True)
        self._loss_masked = parse_bool(cfg_get(cfg, "loss_masked", None), default=True)
        self._normalize = str(cfg_get(cfg, "normalize", "standardize")).strip().lower() or "standardize"
        if self._normalize not in _NORMALIZE_OPTIONS:
            raise ValueError(
                f"decompose.normalize must be one of {_NORMALIZE_OPTIONS}, got {self._normalize}"
            )

        hidden_channels = _parse_int_list(require_cfg(cfg, "hidden_channels", label="decompose"), "hidden_channels")
        self._hidden_channels = hidden_channels
        if not self._hidden_channels:
            raise ValueError("decompose.hidden_channels must be non-empty for autoencoder")

        self._activation = str(cfg_get(cfg, "activation", "relu"))
        self._epochs = int(cfg_get(cfg, "epochs", 25))
        if self._epochs <= 0:
            raise ValueError("decompose.epochs must be > 0 for autoencoder")
        self._batch_size = int(cfg_get(cfg, "batch_size", 16))
        if self._batch_size <= 0:
            raise ValueError("decompose.batch_size must be > 0 for autoencoder")
        self._lr = float(cfg_get(cfg, "lr", 1.0e-3))
        if self._lr <= 0:
            raise ValueError("decompose.lr must be > 0 for autoencoder")
        self._weight_decay = float(cfg_get(cfg, "weight_decay", 0.0))
        if self._weight_decay < 0:
            raise ValueError("decompose.weight_decay must be >= 0 for autoencoder")
        self._device_name = str(cfg_get(cfg, "device", "cpu"))
        self._seed = cfg_get(cfg, "seed", None)
        self._max_train_samples = cfg_get(cfg, "max_train_samples", None)

        self._model: _ConvAutoencoder | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._channels: int | None = None
        self._field_ndim: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._trained_samples: int | None = None
        self._last_loss: float | None = None
        self._torch_device = None
        self._residual_scale: np.ndarray | None = None

    def _ensure_grid_domain(self, domain_spec: DomainSpec) -> None:
        if domain_spec.name == "mesh":
            raise ValueError("autoencoder does not support mesh domains")

    def _resolve_device(self) -> torch.device:
        _require_torch()
        name = self._device_name.lower()
        if name == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("decompose.device=cuda requires available CUDA")
            return torch.device("cuda")
        if name != "cpu":
            raise ValueError("decompose.device must be cpu or cuda for autoencoder")
        return torch.device("cpu")

    def _build_model(self, height: int, width: int, channels: int) -> None:
        if self._model is not None:
            return
        if self._seed is not None:
            torch.manual_seed(int(self._seed))
        self._torch_device = self._resolve_device()
        # CONTRACT: model shape is fixed to the training grid and channel count.
        self._model = _ConvAutoencoder(
            in_channels=channels,
            hidden_channels=self._hidden_channels,
            latent_dim=self._latent_dim,
            height=height,
            width=width,
            activation=self._activation,
        ).to(self._torch_device)

    def _prepare_dataset(
        self,
        dataset: Any,
        domain_spec: DomainSpec,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        fields: list[np.ndarray] = []
        masks: list[np.ndarray | None] = []
        channels: int | None = None
        for idx in range(len(dataset)):
            sample = dataset[idx]
            field = np.asarray(sample.field)
            sample_mask = None if sample.mask is None else np.asarray(sample.mask)
            field_3d, _ = self._prepare_field(
                field,
                sample_mask,
                domain_spec,
                allow_zero_fill=self._mask_policy == "zero_fill",
            )
            if channels is None:
                channels = int(field_3d.shape[-1])
            elif channels != field_3d.shape[-1]:
                raise ValueError("autoencoder requires consistent channel count across samples")
            fields.append(field_3d.astype(np.float32))

            field_mask = _normalize_mask(sample_mask, field_3d.shape[:2])
            combined = _combine_masks(field_mask, domain_spec.mask)
            masks.append(None if combined is None else np.asarray(combined).astype(bool))
        if not fields:
            raise ValueError("autoencoder requires at least one training sample")
        field_arr = np.stack(fields, axis=0)
        if all(mask is None for mask in masks):
            mask_arr = None
        elif any(mask is None for mask in masks):
            raise ValueError("autoencoder requires mask present for all samples or none")
        else:
            mask_arr = np.stack([mask for mask in masks if mask is not None], axis=0)
        if self._max_train_samples is not None:
            max_samples = int(self._max_train_samples)
            if max_samples <= 0:
                raise ValueError("decompose.max_train_samples must be > 0 for autoencoder")
            field_arr = field_arr[:max_samples]
            if mask_arr is not None:
                mask_arr = mask_arr[:max_samples]
        if not np.isfinite(field_arr).all():
            raise ValueError("autoencoder requires finite field values")
        self._channels = channels
        return field_arr, mask_arr

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "AutoencoderDecomposer":
        if dataset is None:
            raise ValueError("autoencoder requires dataset for fit")
        if domain_spec is None:
            raise ValueError("autoencoder requires domain_spec for fit")
        self._ensure_grid_domain(domain_spec)

        field_arr, mask_arr = self._prepare_dataset(dataset, domain_spec)
        height, width = domain_spec.grid_shape
        if field_arr.shape[1:3] != (height, width):
            raise ValueError("autoencoder training grid does not match domain")
        self._grid_shape = (int(height), int(width))
        self._field_ndim = 3
        self._trained_samples = int(field_arr.shape[0])
        channels = int(field_arr.shape[-1])
        coeff_len = int(self._latent_dim) + (channels if self._include_mean else 0)
        self._coeff_shape = (int(coeff_len),)

        self._build_model(height, width, channels)
        if self._model is None or self._torch_device is None:
            raise ValueError("autoencoder model was not initialized")

        # Per-sample mean removal (DC component) improves training on offset-dominant fields.
        means = np.zeros((field_arr.shape[0], channels), dtype=np.float32)
        if self._include_mean:
            for i in range(field_arr.shape[0]):
                m = None if mask_arr is None else mask_arr[i]
                if m is None:
                    means[i] = np.mean(field_arr[i].reshape(-1, channels), axis=0)
                else:
                    if not np.any(m):
                        raise ValueError("autoencoder training mask has no valid entries")
                    for ch in range(channels):
                        means[i, ch] = float(np.mean(field_arr[i, ..., ch][m]))

        residual = field_arr - means[:, None, None, :]
        if mask_arr is not None:
            residual = residual.copy()
            # mask_arr is (N,H,W); boolean indexing with fewer dims selects along spatial
            # axes and preserves the channel axis (sets all channels to 0.0).
            residual[~mask_arr] = 0.0

        scale = np.ones((channels,), dtype=np.float32)
        if self._normalize == "standardize":
            for ch in range(channels):
                if mask_arr is None:
                    var = float(np.mean(residual[..., ch] ** 2))
                else:
                    vals = residual[..., ch][mask_arr]
                    var = float(np.mean(vals**2)) if vals.size else 0.0
                std = float(np.sqrt(max(var, 0.0)))
                if not np.isfinite(std) or std <= 0.0:
                    std = 1.0
                scale[ch] = std
        self._residual_scale = scale
        residual_scaled = residual / scale[None, None, None, :]

        tensor = torch.from_numpy(residual_scaled).permute(0, 3, 1, 2).contiguous()
        mask_tensor = None
        if self._loss_masked and mask_arr is not None:
            mask_tensor = torch.from_numpy(mask_arr.astype(np.float32)).unsqueeze(1).contiguous()

        dataset_t = TensorDataset(tensor, mask_tensor) if mask_tensor is not None else TensorDataset(tensor)
        loader = DataLoader(dataset_t, batch_size=min(self._batch_size, len(dataset_t)), shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        criterion = nn.MSELoss()

        self._model.train()
        last_loss = None
        for _ in range(self._epochs):
            epoch_loss = 0.0
            for batch in loader:
                if mask_tensor is None:
                    (x,) = batch
                    m = None
                else:
                    x, m = batch
                x = x.to(self._torch_device)
                m = None if m is None else m.to(self._torch_device)
                optimizer.zero_grad(set_to_none=True)
                recon = self._model(x)
                if m is None or not self._loss_masked:
                    loss = criterion(recon, x)
                else:
                    diff = recon - x
                    # Broadcast mask over channels.
                    diff2 = (diff * diff) * m
                    denom = torch.clamp(m.sum() * float(diff.shape[1]), min=1.0)
                    loss = diff2.sum() / denom
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * x.shape[0]
            last_loss = epoch_loss / max(1, len(dataset_t))
        self._last_loss = last_loss
        return self

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._ensure_grid_domain(domain_spec)
        if self._model is None or self._torch_device is None:
            raise ValueError("autoencoder fit must be called before transform")
        field_3d, was_2d = self._prepare_field(
            field,
            mask,
            domain_spec,
            allow_zero_fill=self._mask_policy == "zero_fill",
        )
        if self._grid_shape is not None and field_3d.shape[:2] != self._grid_shape:
            raise ValueError("autoencoder field grid does not match fit")
        if self._channels is not None and field_3d.shape[-1] != self._channels:
            raise ValueError("autoencoder field channels do not match fit")

        # Mean removal (mask-aware) to make the latent focus on fluctuations.
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined_mask = _combine_masks(field_mask, domain_spec.mask)
        means = np.zeros((int(field_3d.shape[-1]),), dtype=np.float32)
        if self._include_mean:
            if combined_mask is None:
                means = np.mean(field_3d.reshape(-1, field_3d.shape[-1]), axis=0).astype(np.float32)
            else:
                if not np.any(combined_mask):
                    raise ValueError("autoencoder transform mask has no valid entries")
                for ch in range(int(field_3d.shape[-1])):
                    means[ch] = float(np.mean(np.asarray(field_3d[..., ch])[combined_mask]))

        residual = field_3d.astype(np.float32) - means[None, None, :]
        if combined_mask is not None:
            residual = residual.copy()
            residual[~combined_mask] = 0.0

        scale = np.ones((int(field_3d.shape[-1]),), dtype=np.float32)
        if self._normalize == "standardize":
            if self._residual_scale is None:
                raise ValueError("autoencoder residual scale is missing; call fit first")
            scale = np.asarray(self._residual_scale, dtype=np.float32).reshape(-1)
            if scale.size != int(field_3d.shape[-1]):
                raise ValueError("autoencoder residual scale shape mismatch")
        residual_scaled = residual / scale[None, None, :]

        tensor = torch.from_numpy(residual_scaled).permute(2, 0, 1).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            latent = self._model.encode(tensor.to(self._torch_device))
        latent_vec = latent.cpu().numpy().reshape(-1).astype(np.float32)
        if self._include_mean:
            coeff = np.concatenate([means.reshape(-1), latent_vec], axis=0)
        else:
            coeff = latent_vec

        if self._field_ndim is None:
            self._field_ndim = 2 if was_2d else 3
        self._coeff_shape = (int(coeff.shape[0]),)
        # REVIEW: coeff ordering follows the encoder latent vector.
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [int(field_3d.shape[0]), int(field_3d.shape[1])]
            if was_2d
            else [int(field_3d.shape[0]), int(field_3d.shape[1]), int(field_3d.shape[2])],
            "field_ndim": self._field_ndim,
            "field_layout": "HW" if was_2d else "HWC",
            "channels": int(field_3d.shape[-1]),
            "coeff_shape": [int(coeff.shape[0])],
            "coeff_layout": "K",
            "flatten_order": "C",
            "complex_format": "real",
            "keep": "all",
            "include_mean": bool(self._include_mean),
            "loss_masked": bool(self._loss_masked),
            "normalize": str(self._normalize),
            "residual_scale": None if self._residual_scale is None else [float(x) for x in np.asarray(self._residual_scale).reshape(-1)],
            "latent_dim": int(self._latent_dim),
            "hidden_channels": [int(ch) for ch in self._hidden_channels],
            "activation": self._activation,
            "epochs": int(self._epochs),
            "batch_size": int(self._batch_size),
            "lr": float(self._lr),
            "weight_decay": float(self._weight_decay),
            "mask_policy": self._mask_policy,
            "trained_samples": int(self._trained_samples) if self._trained_samples is not None else None,
            "last_loss": float(self._last_loss) if self._last_loss is not None else None,
            "projection": "conv_autoencoder_residual",
        }
        return coeff

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._ensure_grid_domain(domain_spec)
        if self._model is None or self._torch_device is None:
            raise ValueError("autoencoder fit must be called before inverse_transform")
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("autoencoder transform must be called before inverse_transform")
        if self._grid_shape is not None and domain_spec.grid_shape != self._grid_shape:
            raise ValueError("autoencoder domain grid does not match fit")

        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            padded = np.zeros(expected, dtype=coeff.dtype)
            padded[: coeff.size] = coeff
            coeff = padded

        self._model.eval()
        with torch.no_grad():
            if self._include_mean:
                ch = int(self._channels or 1)
                mean = coeff[:ch].astype(np.float32)
                latent = coeff[ch:].astype(np.float32)
            else:
                mean = np.zeros((int(self._channels or 1),), dtype=np.float32)
                latent = coeff.astype(np.float32)
            z_latent = torch.from_numpy(latent).unsqueeze(0)
            recon = self._model.decode(z_latent.to(self._torch_device))
        residual_scaled = recon.cpu().numpy()[0].transpose(1, 2, 0).astype(np.float32)
        scale = np.ones((residual_scaled.shape[-1],), dtype=np.float32)
        if self._normalize == "standardize":
            if self._residual_scale is None:
                raise ValueError("autoencoder residual scale is missing; call fit first")
            scale = np.asarray(self._residual_scale, dtype=np.float32).reshape(-1)
            if scale.size != residual_scaled.shape[-1]:
                raise ValueError("autoencoder residual scale shape mismatch")
        residual = residual_scaled * scale[None, None, :]
        field = residual + mean[None, None, :]
        if domain_spec.mask is not None:
            field = field.copy()
            field[~domain_spec.mask] = 0.0
        if self._field_ndim == 2 and field.shape[-1] == 1:
            return field[..., 0]
        return field

    def save_state(self, run_dir: str | Path) -> Path:
        path = super().save_state(run_dir)
        if self._model is not None:
            # CONTRACT: torch weights are stored explicitly for reuse.
            out_dir = Path(run_dir) / "outputs" / "states" / "decomposer"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), out_dir / "weights.pt")
        return path
