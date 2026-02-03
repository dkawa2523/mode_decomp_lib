from __future__ import annotations

from pathlib import Path

from mode_decomp_ml.plugins.decomposers import list_decomposers


def test_decomposer_config_coverage() -> None:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs" / "decompose"
    missing = []
    for name in list_decomposers():
        cfg_path = config_dir / f"{name}.yaml"
        if not cfg_path.exists():
            missing.append(str(cfg_path))
    assert not missing, f"Missing decomposer configs: {missing}"
