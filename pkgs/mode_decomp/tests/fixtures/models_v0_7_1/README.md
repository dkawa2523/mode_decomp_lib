# Real 0.7.1 model fixtures

These bundles and expected outputs were produced before the 0.8 refactor, using
the recorded producer runtimes. They are original bundles, not newly generated
models with modified version metadata. Eight fixtures cover DCT, POD/Ridge with
coefficient standardization, calibrated field bands, Hodge, polar FFT,
Laplace-Beltrami, spherical harmonics and spherical Slepian.

The locked environment runs their compatibility checks. A different numerical
backend version skips the historical fixture test because joblib/sklearn model
compatibility across those versions is not supported. New-format roundtrips
remain covered in the current environment.
