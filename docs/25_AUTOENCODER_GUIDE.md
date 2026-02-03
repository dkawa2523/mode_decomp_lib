# Autoencoder decomposer guide

This guide covers practical usage of the `autoencoder` decomposer.

## When to use
- Nonlinear structures not captured by linear bases (POD/FFT/etc.).
- You have enough samples for training and are OK with iterative training.

## Recommended starting config
```
name: autoencoder
mask_policy: zero_fill
latent_dim: 64
hidden_channels: [16, 32]
activation: relu
epochs: 25
batch_size: 16
lr: 1.0e-3
weight_decay: 0.0
device: cpu
seed: 0
```

## Operational guidance
- CPU is fine for small grids (<= 64x64). Use GPU for larger grids or more channels.
- Increase `latent_dim` or `hidden_channels` if reconstructions are too smooth.
- Reduce `epochs` or increase `batch_size` for faster iteration.
- Set `mask_policy: zero_fill` when masks are present to avoid errors.
 - Rule of thumb: GPU recommended when `H*W*channels` exceeds ~64k elements or when `epochs >= 50`.

## Failure modes
- `ImportError: autoencoder decomposer requires torch ...` means torch is missing.
- CUDA device errors mean torch is CPU-only or GPU is unavailable.
- NaNs usually come from too-large learning rate; reduce `lr`.
