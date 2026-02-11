import pickle

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers.pod import PODDecomposer


class _Sample:
    def __init__(self, field: np.ndarray) -> None:
        self.cond = np.zeros((0,), dtype=np.float32)
        self.field = field
        self.mask = None
        self.meta = {}


def test_pod_channelwise_can_be_pickled_after_fit_and_transform() -> None:
    rng = np.random.RandomState(0)
    fields = [rng.randn(8, 8, 2).astype(np.float64) for _ in range(4)]
    dataset = [_Sample(field) for field in fields]

    domain_spec = build_domain_spec({"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}, (8, 8, 2))

    dec = PODDecomposer(
        cfg={"name": "pod", "mask_policy": "error", "backend": "sklearn", "solver": "snapshots", "n_modes": 2}
    )
    dec.fit(dataset=dataset, domain_spec=domain_spec)
    _ = dec.transform(fields[0], mask=None, domain_spec=domain_spec)

    data = pickle.dumps(dec)
    dec2 = pickle.loads(data)
    _ = dec2.transform(fields[1], mask=None, domain_spec=domain_spec)
