"""Process entrypoint: decomposition.

Stage responsibility:
- fit decomposer + codec (and optional preprocess wrapper)
- field -> coeffs (vector) and reconstruction diagnostics
- write standardized artifacts under `run_dir/`
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mode_decomp_ml.data import FieldSample, build_dataset
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.evaluate import compute_metrics, field_r2
from mode_decomp_ml.plugins.codecs import build_coeff_codec
from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.preprocess import build_preprocess
from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_dataset_meta,
    combine_masks,
    cfg_get,
    dataset_to_arrays,
    require_cfg_keys,
    resolve_domain_cfg,
    resolve_run_dir,
    split_indices,
)
from mode_decomp_ml.pipeline.process_base import finalize_run, init_run

from processes._decomposition_metrics import _grid_spacing_from_domain
from processes._decomposition_plots import _render_plots, _resolve_cond_names


class _ArrayDataset:
    def __init__(
        self,
        conds: np.ndarray,
        fields: np.ndarray,
        masks: np.ndarray | None,
        metas: list[dict[str, Any]],
        *,
        name: str,
    ) -> None:
        self._conds = conds
        self._fields = fields
        self._masks = masks
        self._metas = metas
        self.name = name

    def __len__(self) -> int:
        return int(self._conds.shape[0])

    def __getitem__(self, index: int) -> FieldSample:
        idx = int(index)
        return FieldSample(
            cond=self._conds[idx],
            field=self._fields[idx],
            mask=None if self._masks is None else self._masks[idx],
            meta=self._metas[idx],
        )

def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("decomposition requires config from the Hydra entrypoint")
    require_cfg_keys(
        cfg,
        ["seed", "run_dir", "dataset", "split", "domain", "decompose", "codec", "preprocess", "eval", "viz"],
    )

    run_dir = resolve_run_dir(cfg)
    writer = ArtifactWriter(run_dir)
    steps = StepRecorder(run_dir=run_dir)
    init_run(writer=writer, steps=steps, cfg=cfg, run_dir=run_dir, clean=True, snapshot=True)

    seed = cfg_get(cfg, "seed", None)
    dataset_cfg = cfg_get(cfg, "dataset")
    domain_cfg = cfg_get(cfg, "domain")
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, domain_cfg)
    split_cfg = cfg_get(cfg, "split")

    cond_names: list[str] | None = None

    with steps.step(
        "build_dataset",
        inputs=[artifact_ref("configuration/run.yaml", kind="config")],
    ) as step:
        dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=seed)
        conds, fields, masks, metas = dataset_to_arrays(dataset)
        split_meta = split_indices(split_cfg, len(dataset), seed)
        idx = np.asarray(split_meta["train_idx"], dtype=int)
        conds = conds[idx]
        fields = fields[idx]
        masks = masks[idx] if masks is not None else None
        metas = [metas[i] for i in idx.tolist()]
        step["meta"]["dataset_name"] = getattr(dataset, "name", None)
        step["meta"]["num_samples"] = int(len(conds))
        cond_names = _resolve_cond_names(dataset, conds)

    with steps.step(
        "preprocess_fields",
        outputs=[artifact_ref("outputs/states/preprocess/state.pkl", kind="state")],
    ):
        preprocess = build_preprocess(cfg_get(cfg, "preprocess"))
        preprocess.fit(fields, masks, split="train")
        fields_proc, masks_proc = preprocess.transform(fields, masks)
        preprocess.save_state(run_dir)

    domain_spec = build_domain_spec(domain_cfg, fields_proc.shape[1:])
    decompose_cfg = cfg_get(cfg, "decompose", {}) or {}
    offset_split_cfg = cfg_get(cfg, "offset_split", None)
    use_offset_split = False
    if isinstance(offset_split_cfg, Mapping):
        enabled = cfg_get(offset_split_cfg, "enabled", "false")
        if isinstance(enabled, bool):
            use_offset_split = enabled
        elif isinstance(enabled, (int, np.integer)):
            use_offset_split = bool(int(enabled))
        else:
            enabled_str = str(enabled).strip().lower()
            use_offset_split = enabled_str not in {"", "false", "0", "no", "n"}
    if use_offset_split:
        decompose_cfg_effective = {"name": "offset_residual", "inner": dict(decompose_cfg)}
        # Pass through offset_split parameters to the wrapper decomposer.
        decompose_cfg_effective.update(dict(offset_split_cfg))
        decomposer = build_decomposer(decompose_cfg_effective)
    else:
        decomposer = build_decomposer(decompose_cfg)
    codec = build_coeff_codec(cfg_get(cfg, "codec"))

    fit_time_sec: float | None = None
    with steps.step(
        "fit_decomposer",
        outputs=[artifact_ref("outputs/states/decomposer/state.pkl", kind="state")],
    ) as step:
        start = time.perf_counter()
        decomposer.fit(
            dataset=_ArrayDataset(conds, fields_proc, masks_proc, metas, name=dataset.name),
            domain_spec=domain_spec,
        )
        fit_time_sec = float(time.perf_counter() - start)
        step["meta"]["fit_time_sec"] = fit_time_sec

    coeffs = []
    raw_meta: Mapping[str, Any] | None = None
    sample_ids: list[str] = []
    with steps.step(
        "encode_coeffs",
        outputs=[
            artifact_ref("outputs/states/decomposer/coeff_meta.json", kind="metadata"),
            artifact_ref("outputs/states/decomposer/state.pkl", kind="state"),
            artifact_ref("outputs/states/coeff_meta.json", kind="metadata"),
            artifact_ref("outputs/states/coeff_codec/coeff_meta.json", kind="metadata"),
            artifact_ref("outputs/states/coeff_codec/state.pkl", kind="state"),
            artifact_ref("outputs/coeffs.npz", kind="coeffs"),
        ],
    ):
        for idx in range(fields_proc.shape[0]):
            raw_coeff = decomposer.transform(
                fields_proc[idx],
                mask=None if masks_proc is None else masks_proc[idx],
                domain_spec=domain_spec,
            )
            if raw_meta is None:
                raw_meta = decomposer.coeff_meta()
            if isinstance(raw_coeff, Mapping) and getattr(codec, "name", "") not in {
                "auto_codec_v1",
                "offset_residual_pack_v1",
            }:
                raise ValueError(
                    "offset_split produced mapping coefficients; please set "
                    "codec.name=auto_codec_v1 (recommended) or offset_residual_pack_v1"
                )
            coeffs.append(codec.encode(raw_coeff, raw_meta))
        A = np.stack(coeffs, axis=0)
        if raw_meta is None:
            raise ValueError("coeff_meta missing after decomposer.transform")

        decomposer.save_coeff_meta(run_dir)
        decomposer.save_state(run_dir)
        codec.save_coeff_meta(run_dir, raw_meta, A[0])
        codec.save_state(run_dir)
        sample_ids = [meta.get("sample_id", f"sample_{i:04d}") for i, meta in enumerate(metas)]
        writer.write_coeffs(
            {
                "coeff": A,
                "cond": conds,
                "sample_ids": np.asarray(sample_ids, dtype=object),
            }
        )

    with steps.step(
        "reconstruct_field",
        outputs=[artifact_ref("outputs/preds.npz", kind="preds")],
    ):
        fields_recon = []
        for idx in range(A.shape[0]):
            raw_coeff = codec.decode(A[idx], raw_meta)
            fields_recon.append(decomposer.inverse_transform(raw_coeff, domain_spec=domain_spec))
        field_hat = np.stack(fields_recon, axis=0)
        field_hat, _ = preprocess.inverse_transform(field_hat, None)
        writer.write_preds({"field": field_hat})

    eval_cfg = cfg_get(cfg, "eval", {}) or {}
    metrics_list = list(cfg_get(eval_cfg, "metrics", []))
    if not metrics_list:
        raise ValueError("eval.metrics must be configured for decomposition")

    # Always evaluate on the combined (dataset ∩ domain) mask when either is present.
    # This avoids silently including outside-domain pixels (e.g. annulus) in metrics.
    masks_eval = combine_masks(
        masks,
        domain_spec.mask,
        spatial_shape=fields.shape[1:3],
        n_samples=fields.shape[0],
    )

    needs_divcurl = any(name in {"div_rmse", "curl_rmse"} for name in metrics_list)
    grid_spacing = _grid_spacing_from_domain(domain_spec) if needs_divcurl else None

    coeff_meta = codec.coeff_meta(raw_meta, A[0])
    with steps.step(
        "compute_metrics",
        outputs=[artifact_ref("outputs/metrics.json", kind="metrics")],
    ):
        metrics = compute_metrics(
            metrics_list,
            field_true=fields,
            field_pred=field_hat,
            mask=masks_eval,
            coeff_true_a=A,
            coeff_pred_a=A,
            coeff_meta=coeff_meta,
            grid_spacing=grid_spacing,
        )
        # Persist offset split decision, when available.
        try:
            if isinstance(raw_meta, Mapping):
                if "split_enabled" in raw_meta:
                    metrics["offset_split_enabled"] = bool(raw_meta.get("split_enabled"))
                elif "offset_split_enabled" in raw_meta:
                    metrics["offset_split_enabled"] = bool(raw_meta.get("offset_split_enabled"))
                if "split_ratio_median" in raw_meta:
                    metrics["offset_ratio_median"] = float(raw_meta.get("split_ratio_median"))
                elif "offset_ratio_median" in raw_meta:
                    metrics["offset_ratio_median"] = float(raw_meta.get("offset_ratio_median"))
                if "f_offset" in raw_meta:
                    metrics["f_offset"] = float(raw_meta.get("f_offset"))
        except Exception:
            pass
        # Additional "R^2 @ K" diagnostics: reconstruct using only a subset of coefficients.
        # This is intentionally simple and only supports common layouts:
        #   - CK: keep the first K modes per channel
        #   - CHW: keep a low-frequency block where s=ceil(sqrt(K)) per channel
        try:
            coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()

            def _reshape_if_vector(arr: np.ndarray, shape: tuple[int, ...], order: str) -> np.ndarray:
                if arr.ndim != 1:
                    return arr
                return arr.reshape(shape, order=order)

            def _truncate_ck(arr: np.ndarray, *, k_keep: int) -> np.ndarray:
                out = np.asarray(arr).copy()
                if out.ndim != 2:
                    raise ValueError("CK truncate requires 2D array")
                if k_keep < out.shape[1]:
                    out[:, k_keep:] = 0.0
                return out

            def _truncate_chw(arr: np.ndarray, *, k_keep: int, height: int, width: int) -> np.ndarray:
                out = np.zeros_like(arr)
                s = int(np.ceil(np.sqrt(k_keep)))
                s_h = min(s, height)
                s_w = min(s, width)
                out[:, :s_h, :s_w] = arr[:, :s_h, :s_w]
                return out

            def _truncate_fft2_chw(arr: np.ndarray, *, k_keep: int, height: int, width: int, fft_shift: bool) -> np.ndarray:
                # Keep a centered low-frequency block in the (shifted) frequency domain.
                s = int(np.ceil(np.sqrt(k_keep)))
                s_h = min(max(1, s), height)
                s_w = min(max(1, s), width)
                if fft_shift:
                    shifted = np.asarray(arr)
                else:
                    shifted = np.fft.fftshift(np.asarray(arr), axes=(1, 2))
                out_shifted = np.zeros_like(shifted)
                cy = int(height // 2)
                cx = int(width // 2)
                y0 = int(cy - (s_h // 2))
                x0 = int(cx - (s_w // 2))
                y0 = max(0, min(y0, height - s_h))
                x0 = max(0, min(x0, width - s_w))
                out_shifted[:, y0 : y0 + s_h, x0 : x0 + s_w] = shifted[:, y0 : y0 + s_h, x0 : x0 + s_w]
                if fft_shift:
                    return out_shifted
                return np.fft.ifftshift(out_shifted, axes=(1, 2))

            if coeff_format == "offset_residual_v1":
                residual_raw_meta = raw_meta.get("residual_raw_meta")
                if not isinstance(residual_raw_meta, Mapping):
                    raise ValueError("offset_residual field_r2_k requires residual_raw_meta")
                res_layout = str(residual_raw_meta.get("coeff_layout", "")).strip().upper()
                res_shape = residual_raw_meta.get("coeff_shape")
                res_complex = str(residual_raw_meta.get("complex_format", "")).strip().lower()
                res_method = str(residual_raw_meta.get("method", "")).strip().lower()
                allow_complex_chw = res_method in {"fft2", "fft2_lowpass"} and res_layout == "CHW" and res_complex == "complex"
                if res_complex != "real" and not allow_complex_chw:
                    raise ValueError("offset_residual field_r2_k supports real residual coefficients only")
                if not isinstance(res_shape, (list, tuple)):
                    raise ValueError("residual coeff_shape missing for field_r2_k metrics")
                order = str(residual_raw_meta.get("flatten_order", "C")).strip().upper() or "C"
                k_eval_list = [1, 4, 16, 64]
                if res_layout == "CK" and len(res_shape) == 2:
                    k_total = int(res_shape[1])
                    for k_eval in k_eval_list:
                        k_keep = min(int(k_eval), max(k_total, 0))
                        if k_keep <= 0:
                            continue
                        fields_k = []
                        for idx in range(A.shape[0]):
                            decoded = codec.decode(A[idx], raw_meta)
                            if not isinstance(decoded, Mapping):
                                raise ValueError("offset_residual decode did not return a mapping")
                            offset = decoded.get("offset")
                            residual = decoded.get("residual")
                            if offset is None or residual is None:
                                raise ValueError("offset_residual decode missing offset/residual")
                            arr = _reshape_if_vector(np.asarray(residual), (int(res_shape[0]), int(res_shape[1])), order)
                            arr_k = _truncate_ck(arr, k_keep=k_keep)
                            fields_k.append(
                                decomposer.inverse_transform({"offset": offset, "residual": arr_k}, domain_spec=domain_spec)
                            )
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)
                elif res_layout == "CHW" and len(res_shape) == 3:
                    height_c = int(res_shape[1])
                    width_c = int(res_shape[2])
                    if height_c <= 0 or width_c <= 0:
                        raise ValueError("invalid residual CHW coeff shape for field_r2_k metrics")
                    fft_shift = bool(residual_raw_meta.get("fft_shift", False))
                    for k_eval in k_eval_list:
                        k_keep = int(k_eval)
                        if k_keep <= 0:
                            continue
                        fields_k = []
                        for idx in range(A.shape[0]):
                            decoded = codec.decode(A[idx], raw_meta)
                            if not isinstance(decoded, Mapping):
                                raise ValueError("offset_residual decode did not return a mapping")
                            offset = decoded.get("offset")
                            residual = decoded.get("residual")
                            if offset is None or residual is None:
                                raise ValueError("offset_residual decode missing offset/residual")
                            arr = _reshape_if_vector(
                                np.asarray(residual),
                                (int(res_shape[0]), height_c, width_c),
                                order,
                            )
                            if arr.ndim != 3:
                                raise ValueError("residual CHW truncate requires 3D array")
                            if allow_complex_chw:
                                arr_k = _truncate_fft2_chw(
                                    arr,
                                    k_keep=k_keep,
                                    height=height_c,
                                    width=width_c,
                                    fft_shift=fft_shift,
                                )
                            else:
                                arr_k = _truncate_chw(arr, k_keep=k_keep, height=height_c, width=width_c)
                            fields_k.append(
                                decomposer.inverse_transform({"offset": offset, "residual": arr_k}, domain_spec=domain_spec)
                            )
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)
                elif res_layout == "K" and len(res_shape) == 1:
                    k_total = int(res_shape[0])
                    for k_eval in k_eval_list:
                        k_keep = min(int(k_eval), max(k_total, 0))
                        if k_keep <= 0:
                            continue
                        fields_k = []
                        for idx in range(A.shape[0]):
                            decoded = codec.decode(A[idx], raw_meta)
                            if not isinstance(decoded, Mapping):
                                raise ValueError("offset_residual decode did not return a mapping")
                            offset = decoded.get("offset")
                            residual = decoded.get("residual")
                            if offset is None or residual is None:
                                raise ValueError("offset_residual decode missing offset/residual")
                            vec = np.asarray(residual).reshape(-1)
                            if vec.size != k_total:
                                raise ValueError("residual K coeff shape mismatch for field_r2_k metrics")
                            vec_k = vec.copy()
                            if k_keep < vec_k.size:
                                vec_k[k_keep:] = 0.0
                            fields_k.append(
                                decomposer.inverse_transform({"offset": offset, "residual": vec_k}, domain_spec=domain_spec)
                            )
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)
                else:
                    raise ValueError("unsupported residual coeff layout for field_r2_k")
            else:
                raw_coeff_layout = str(raw_meta.get("coeff_layout", "")).strip().upper()
                raw_coeff_shape = raw_meta.get("coeff_shape")
                raw_complex = str(raw_meta.get("complex_format", "")).strip().lower()
                raw_method = str(raw_meta.get("method", "")).strip().lower()
                allow_fft2_complex = raw_method in {"fft2", "fft2_lowpass"} and raw_coeff_layout == "CHW" and raw_complex == "complex"
                if raw_complex != "real" and not allow_fft2_complex:
                    raise ValueError("field_r2_k metrics only support real coefficients")
                if not isinstance(raw_coeff_shape, (list, tuple)):
                    raise ValueError("coeff_shape missing for field_r2_k metrics")
                if raw_coeff_layout == "CK" and len(raw_coeff_shape) == 2:
                    k_total = int(raw_coeff_shape[1])
                    k_eval_list = [1, 4, 16, 64]
                    for k_eval in k_eval_list:
                        k_eval = int(k_eval)
                        if k_eval <= 0:
                            continue
                        if k_total <= 0:
                            continue
                        k_keep = min(k_eval, k_total)
                        fields_k = []
                        for idx in range(A.shape[0]):
                            raw_coeff = codec.decode(A[idx], raw_meta)
                            arr = np.asarray(raw_coeff)
                            if arr.ndim == 1:
                                arr = arr.reshape((int(raw_coeff_shape[0]), int(raw_coeff_shape[1])), order="C")
                            if arr.ndim != 2:
                                raise ValueError("field_r2_k metrics require 2D CK coeff array")
                            arr_k = arr.copy()
                            if k_keep < arr_k.shape[1]:
                                arr_k[:, k_keep:] = 0.0
                            fields_k.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)
                elif raw_coeff_layout == "CHW" and len(raw_coeff_shape) == 3:
                    height_c = int(raw_coeff_shape[1])
                    width_c = int(raw_coeff_shape[2])
                    if height_c <= 0 or width_c <= 0:
                        raise ValueError("invalid CHW coeff shape for field_r2_k metrics")
                    k_eval_list = [1, 4, 16, 64]
                    fft_shift = bool(raw_meta.get("fft_shift", False))
                    for k_eval in k_eval_list:
                        k_eval = int(k_eval)
                        if k_eval <= 0:
                            continue
                        fields_k = []
                        for idx in range(A.shape[0]):
                            raw_coeff = codec.decode(A[idx], raw_meta)
                            arr = np.asarray(raw_coeff)
                            if arr.ndim == 1:
                                arr = arr.reshape((int(raw_coeff_shape[0]), height_c, width_c), order="C")
                            if arr.ndim != 3:
                                raise ValueError("field_r2_k metrics require 3D CHW coeff array")
                            if allow_fft2_complex:
                                arr_k = _truncate_fft2_chw(
                                    arr,
                                    k_keep=int(k_eval),
                                    height=height_c,
                                    width=width_c,
                                    fft_shift=fft_shift,
                                )
                            else:
                                # Keep a square low-frequency block (DCT-like ordering).
                                s = int(np.ceil(np.sqrt(k_eval)))
                                s_h = min(s, height_c)
                                s_w = min(s, width_c)
                                arr_k = np.zeros_like(arr)
                                arr_k[:, :s_h, :s_w] = arr[:, :s_h, :s_w]
                            fields_k.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)
                elif raw_coeff_layout == "K" and len(raw_coeff_shape) == 1:
                    k_total = int(raw_coeff_shape[0])
                    k_eval_list = [1, 4, 16, 64]
                    for k_eval in k_eval_list:
                        k_eval = int(k_eval)
                        if k_eval <= 0:
                            continue
                        if k_total <= 0:
                            continue
                        k_keep = min(k_eval, k_total)
                        fields_k = []
                        for idx in range(A.shape[0]):
                            raw_coeff = codec.decode(A[idx], raw_meta)
                            arr = np.asarray(raw_coeff).reshape(-1)
                            if arr.size != k_total:
                                raise ValueError("field_r2_k metrics require 1D K coeff array")
                            arr_k = arr.copy()
                            if k_keep < arr_k.size:
                                arr_k[k_keep:] = 0.0
                            fields_k.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)
        except Exception:
            # Diagnostic only; keep core metrics robust.
            pass
        # "Top-energy @ K" diagnostics for CK layouts:
        # Some decomposers (e.g. RBF centers, wavelet approx) do not have a meaningful
        # coefficient ordering for prefix truncation. This provides a more comparable
        # "best-K" reconstruction by keeping the K modes with the largest mean energy.
        try:
            coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()
            k_eval_list = [1, 4, 16, 64]
            k_grid = [1, 2, 4, 8, 16, 32, 64]
            r2_threshold = 0.95

            def _reshape_if_vector(arr: np.ndarray, shape: tuple[int, ...], order: str) -> np.ndarray:
                if arr.ndim != 1:
                    return arr
                return arr.reshape(shape, order=order)

            def _topk_order_ck(arr_list: list[np.ndarray]) -> np.ndarray:
                k_total = int(arr_list[0].shape[1])
                energy = np.zeros((k_total,), dtype=np.float64)
                for a2 in arr_list:
                    energy += np.sum(a2.astype(np.float64) ** 2, axis=0)
                return np.argsort(-energy)  # desc

            if coeff_format == "offset_residual_v1":
                residual_raw_meta = raw_meta.get("residual_raw_meta")
                if not isinstance(residual_raw_meta, Mapping):
                    raise ValueError("offset_residual topk requires residual_raw_meta")
                layout = str(residual_raw_meta.get("coeff_layout", "")).strip().upper()
                coeff_shape = residual_raw_meta.get("coeff_shape")
                complex_format = str(residual_raw_meta.get("complex_format", "")).strip().lower()
                order = str(residual_raw_meta.get("flatten_order", "C")).strip().upper() or "C"
                if complex_format != "real":
                    raise ValueError("offset_residual topk supports real residual coefficients only")
                if not isinstance(coeff_shape, (list, tuple)) or not coeff_shape:
                    raise ValueError("offset_residual topk requires residual coeff_shape")

                if layout == "CK":
                    if len(coeff_shape) != 2:
                        raise ValueError("offset_residual topk requires residual coeff_shape [C,K]")
                    c = int(coeff_shape[0])
                    k_total = int(coeff_shape[1])
                    if k_total <= 0 or c <= 0:
                        raise ValueError("offset_residual topk invalid coeff_shape")

                    offsets: list[Any] = []
                    residuals: list[np.ndarray] = []
                    for idx in range(A.shape[0]):
                        decoded = codec.decode(A[idx], raw_meta)
                        if not isinstance(decoded, Mapping):
                            raise ValueError("offset_residual decode did not return a mapping")
                        offset = decoded.get("offset")
                        residual = decoded.get("residual")
                        if offset is None or residual is None:
                            raise ValueError("offset_residual decode missing offset/residual")
                        arr = _reshape_if_vector(np.asarray(residual), (c, k_total), order)
                        if arr.ndim != 2 or arr.shape != (c, k_total):
                            raise ValueError("offset_residual residual shape mismatch")
                        offsets.append(offset)
                        residuals.append(arr)

                    order_idx = _topk_order_ck(residuals)
                    for k_eval in k_eval_list:
                        k_keep = min(int(k_eval), k_total)
                        if k_keep <= 0:
                            continue
                        keep = order_idx[:k_keep]
                        fields_k = []
                        for offset, arr in zip(offsets, residuals):
                            arr_k = np.zeros_like(arr)
                            arr_k[:, keep] = arr[:, keep]
                            fields_k.append(
                                decomposer.inverse_transform({"offset": offset, "residual": arr_k}, domain_spec=domain_spec)
                            )
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_topk_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)

                    k_req: int | None = None
                    for k_eval in k_grid:
                        k_keep = min(int(k_eval), k_total)
                        if k_keep <= 0:
                            continue
                        keep = order_idx[:k_keep]
                        fields_k = []
                        for offset, arr in zip(offsets, residuals):
                            arr_k = np.zeros_like(arr)
                            arr_k[:, keep] = arr[:, keep]
                            fields_k.append(
                                decomposer.inverse_transform({"offset": offset, "residual": arr_k}, domain_spec=domain_spec)
                            )
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        r2_val = float(field_r2(fields, field_k, mask=masks_eval))
                        if np.isfinite(r2_val) and r2_val >= float(r2_threshold):
                            k_req = int(k_keep)
                            break
                    if k_req is not None:
                        metrics["k_req_r2_0p95"] = int(k_req)
                elif layout == "K":
                    if len(coeff_shape) != 1:
                        raise ValueError("offset_residual topk requires residual coeff_shape [K]")
                    k_total = int(coeff_shape[0])
                    if k_total <= 0:
                        raise ValueError("offset_residual topk invalid K coeff_shape")

                    offsets: list[Any] = []
                    residuals: list[np.ndarray] = []
                    for idx in range(A.shape[0]):
                        decoded = codec.decode(A[idx], raw_meta)
                        if not isinstance(decoded, Mapping):
                            raise ValueError("offset_residual decode did not return a mapping")
                        offset = decoded.get("offset")
                        residual = decoded.get("residual")
                        if offset is None or residual is None:
                            raise ValueError("offset_residual decode missing offset/residual")
                        vec = np.asarray(residual).reshape(-1)
                        if vec.size != k_total:
                            raise ValueError("offset_residual residual K shape mismatch")
                        offsets.append(offset)
                        residuals.append(vec)

                    energy = np.zeros((k_total,), dtype=np.float64)
                    for vec in residuals:
                        energy += (vec.astype(np.float64) ** 2)
                    order_idx = np.argsort(-energy)  # desc

                    for k_eval in k_eval_list:
                        k_keep = min(int(k_eval), k_total)
                        if k_keep <= 0:
                            continue
                        keep = order_idx[:k_keep]
                        fields_k = []
                        for offset, vec in zip(offsets, residuals):
                            vec_k = np.zeros_like(vec)
                            vec_k[keep] = vec[keep]
                            fields_k.append(
                                decomposer.inverse_transform({"offset": offset, "residual": vec_k}, domain_spec=domain_spec)
                            )
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_topk_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)

                    k_req: int | None = None
                    for k_eval in k_grid:
                        k_keep = min(int(k_eval), k_total)
                        if k_keep <= 0:
                            continue
                        keep = order_idx[:k_keep]
                        fields_k = []
                        for offset, vec in zip(offsets, residuals):
                            vec_k = np.zeros_like(vec)
                            vec_k[keep] = vec[keep]
                            fields_k.append(
                                decomposer.inverse_transform({"offset": offset, "residual": vec_k}, domain_spec=domain_spec)
                            )
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        r2_val = float(field_r2(fields, field_k, mask=masks_eval))
                        if np.isfinite(r2_val) and r2_val >= float(r2_threshold):
                            k_req = int(k_keep)
                            break
                    if k_req is not None:
                        metrics["k_req_r2_0p95"] = int(k_req)
                else:
                    raise ValueError("offset_residual topk supports CK or K residual only")
            else:
                layout = str(raw_meta.get("coeff_layout", "")).strip().upper()
                coeff_shape = raw_meta.get("coeff_shape")
                complex_format = str(raw_meta.get("complex_format", "")).strip().lower()
                order = str(raw_meta.get("flatten_order", "C")).strip().upper() or "C"
                if complex_format != "real":
                    raise ValueError("topk supports real coefficients only")
                if not isinstance(coeff_shape, (list, tuple)) or not coeff_shape:
                    raise ValueError("topk requires coeff_shape")

                if layout == "CK":
                    if len(coeff_shape) != 2:
                        raise ValueError("topk requires coeff_shape [C,K]")
                    c = int(coeff_shape[0])
                    k_total = int(coeff_shape[1])
                    if k_total <= 0 or c <= 0:
                        raise ValueError("topk invalid coeff_shape")

                    coeffs_ck: list[np.ndarray] = []
                    for idx in range(A.shape[0]):
                        raw_coeff = codec.decode(A[idx], raw_meta)
                        if isinstance(raw_coeff, Mapping):
                            raise ValueError("topk requires numeric CK coefficients")
                        arr = _reshape_if_vector(np.asarray(raw_coeff), (c, k_total), order)
                        if arr.ndim != 2 or arr.shape != (c, k_total):
                            raise ValueError("topk CK shape mismatch")
                        coeffs_ck.append(arr)

                    order_idx = _topk_order_ck(coeffs_ck)
                    for k_eval in k_eval_list:
                        k_keep = min(int(k_eval), k_total)
                        if k_keep <= 0:
                            continue
                        keep = order_idx[:k_keep]
                        fields_k = []
                        for arr in coeffs_ck:
                            arr_k = np.zeros_like(arr)
                            arr_k[:, keep] = arr[:, keep]
                            fields_k.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_topk_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)

                    k_req: int | None = None
                    for k_eval in k_grid:
                        k_keep = min(int(k_eval), k_total)
                        if k_keep <= 0:
                            continue
                        keep = order_idx[:k_keep]
                        fields_k = []
                        for arr in coeffs_ck:
                            arr_k = np.zeros_like(arr)
                            arr_k[:, keep] = arr[:, keep]
                            fields_k.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        r2_val = float(field_r2(fields, field_k, mask=masks_eval))
                        if np.isfinite(r2_val) and r2_val >= float(r2_threshold):
                            k_req = int(k_keep)
                            break
                    if k_req is not None:
                        metrics["k_req_r2_0p95"] = int(k_req)
                elif layout == "K":
                    if len(coeff_shape) != 1:
                        raise ValueError("topk requires coeff_shape [K]")
                    k_total = int(coeff_shape[0])
                    if k_total <= 0:
                        raise ValueError("topk invalid K coeff_shape")

                    coeffs_k: list[np.ndarray] = []
                    for idx in range(A.shape[0]):
                        raw_coeff = codec.decode(A[idx], raw_meta)
                        if isinstance(raw_coeff, Mapping):
                            raise ValueError("topk requires numeric K coefficients")
                        vec = np.asarray(raw_coeff).reshape(-1)
                        if vec.size != k_total:
                            raise ValueError("topk K shape mismatch")
                        coeffs_k.append(vec)

                    energy = np.zeros((k_total,), dtype=np.float64)
                    for vec in coeffs_k:
                        energy += (vec.astype(np.float64) ** 2)
                    order_idx = np.argsort(-energy)  # desc

                    for k_eval in k_eval_list:
                        k_keep = min(int(k_eval), k_total)
                        if k_keep <= 0:
                            continue
                        keep = order_idx[:k_keep]
                        fields_k = []
                        for vec in coeffs_k:
                            vec_k = np.zeros_like(vec)
                            vec_k[keep] = vec[keep]
                            fields_k.append(decomposer.inverse_transform(vec_k, domain_spec=domain_spec))
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        metrics[f"field_r2_topk_k{k_eval}"] = field_r2(fields, field_k, mask=masks_eval)

                    k_req: int | None = None
                    for k_eval in k_grid:
                        k_keep = min(int(k_eval), k_total)
                        if k_keep <= 0:
                            continue
                        keep = order_idx[:k_keep]
                        fields_k = []
                        for vec in coeffs_k:
                            vec_k = np.zeros_like(vec)
                            vec_k[keep] = vec[keep]
                            fields_k.append(decomposer.inverse_transform(vec_k, domain_spec=domain_spec))
                        field_k = np.stack(fields_k, axis=0)
                        field_k, _ = preprocess.inverse_transform(field_k, None)
                        r2_val = float(field_r2(fields, field_k, mask=masks_eval))
                        if np.isfinite(r2_val) and r2_val >= float(r2_threshold):
                            k_req = int(k_keep)
                            break
                    if k_req is not None:
                        metrics["k_req_r2_0p95"] = int(k_req)
                else:
                    raise ValueError("topk supports CK or K only")
        except Exception:
            pass
        if fit_time_sec is not None:
            metrics["fit_time_sec"] = fit_time_sec
        writer.write_metrics(metrics)

    with steps.step(
        "render_plots",
        outputs=[artifact_ref("plots", kind="plots_dir")],
    ):
        _render_plots(
            writer=writer,
            fields_true=fields,
            fields_pred=field_hat,
            masks=masks_eval,
            coeffs=A,
            coeff_meta=coeff_meta,
            decomposer=decomposer,
            domain_spec=domain_spec,
            domain_cfg=domain_cfg,
            viz_cfg=cfg_get(cfg, "viz", {}) or {},
            conds=conds,
            cond_names=cond_names,
            codec=codec,
            raw_meta=raw_meta,
            preprocess=preprocess,
            sample_ids=sample_ids,
        )

    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)
    finalize_run(
        writer=writer,
        steps=steps,
        cfg=cfg,
        dataset_hash=str(dataset_meta.get("dataset_hash")) if dataset_meta else None,
        dataset_meta=dataset_meta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
