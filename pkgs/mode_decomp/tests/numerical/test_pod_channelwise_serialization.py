from ml_platform_core.io import dump_joblib, load_joblib

import numpy as np

from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.decomposition.pod import PODDecomposer


class _Sample:
    def __init__(self, field: np.ndarray) -> None:
        self.cond = np.zeros((0,), dtype=np.float32)
        self.field = field
        self.mask = None
        self.meta = {}


def test_pod_channelwise_survives_core_joblib(tmp_path) -> None:
    rng = np.random.RandomState(0)
    fields = [rng.randn(8, 8, 2).astype(np.float64) for _ in range(4)]
    dataset = [_Sample(field) for field in fields]

    domain_spec = build_domain_spec({"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}, (8, 8, 2))

    dec = PODDecomposer(
        cfg={"name": "pod", "mask_policy": "error", "backend": "sklearn", "solver": "snapshots", "n_modes": 2}
    )
    dec.fit(dataset=dataset, domain_spec=domain_spec)
    _ = dec.transform(fields[0], mask=None, domain_spec=domain_spec)

    expected = dec.transform(fields[1], mask=None, domain_spec=domain_spec)
    expected_field = dec.inverse_transform(expected, domain_spec=domain_spec)
    path = dump_joblib(dec, tmp_path / "decomposer.joblib")
    restored = load_joblib(path)
    actual = restored.transform(fields[1], mask=None, domain_spec=domain_spec)
    assert np.allclose(actual, expected)
    assert np.allclose(restored.inverse_transform(actual, domain_spec=domain_spec), expected_field)
