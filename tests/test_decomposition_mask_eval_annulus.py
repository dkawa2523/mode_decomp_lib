from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.evaluate import field_rmse


def test_decomposition_metrics_use_domain_mask_for_annulus(tmp_path: Path, monkeypatch) -> None:
    # Create a minimal npy_dir dataset with a constant field on the full grid.
    # Annular decomposers typically reconstruct 0 outside the annulus mask; metrics must ignore outside.
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    n = 2
    h = 33
    w = 33
    cond = np.zeros((n, 1), dtype=np.float32)
    field = np.ones((n, h, w, 1), dtype=np.float32)
    np.save(dataset_root / "cond.npy", cond)
    np.save(dataset_root / "field.npy", field)

    # Disable plotting for this integration test.
    from processes import decomposition as decomposition_process

    monkeypatch.setattr(decomposition_process, "_render_plots", lambda **_kwargs: None)

    run_dir = tmp_path / "run"
    domain_cfg = {
        "name": "annulus",
        "center": [0.0, 0.0],
        "r_inner": 0.4,
        "r_outer": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }

    cfg = {
        "seed": 0,
        "task": {"name": "decomposition"},
        "run_dir": str(run_dir),
        "dataset": {"name": "npy_dir", "root": str(dataset_root), "mask_policy": "allow_none"},
        "split": {"name": "all"},
        "domain": domain_cfg,
        "decompose": {
            "name": "annular_zernike",
            "n_max": 2,
            "m_max": 2,
            "ordering": "n_then_m",
            "normalization": "orthonormal",
            "boundary_condition": "unit_annulus",
        },
        "codec": {"name": "none"},
        "preprocess": {"name": "basic"},
        "eval": {"name": "basic", "metrics": ["field_rmse"]},
        "viz": {"name": "basic"},
    }

    decomposition_process.main(cfg)

    metrics_path = run_dir / "outputs" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    got_rmse = float(metrics["field_rmse"])

    preds = np.load(run_dir / "outputs" / "preds.npz")
    field_hat = np.asarray(preds["field"])

    domain = build_domain_spec(domain_cfg, (h, w, 1))
    assert domain.mask is not None
    rmse_masked = field_rmse(field, field_hat, mask=np.broadcast_to(domain.mask, (n, h, w)))
    rmse_full = field_rmse(field, field_hat, mask=None)

    # The metric must match the masked evaluation and differ from full-grid evaluation.
    assert abs(got_rmse - rmse_masked) < 1e-6
    assert rmse_full > rmse_masked + 0.05
