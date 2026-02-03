# Optional dependencies

This repository includes components that rely on optional third-party libraries. If you do not need a component, choose another decomposer/model or remove it from your config.

## Decomposers
- wavelet2d: `pywt`
- pswf2d_tensor: `scipy`
- spherical_harmonics: `pyshtools`
- spherical_slepian: `pyshtools`
- autoencoder: `torch`
- pod (modred backend): `modred`

## Models
- xgb: `xgboost`
- lgbm: `lightgbm`
- catboost: `catboost`
- mtgp: `gpytorch`
